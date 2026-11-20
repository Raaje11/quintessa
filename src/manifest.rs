//! Capability Lowering & Feature Neuron Manifest.
//!
//! Bridges `CapabilityGraph` (what domain capabilities are required) and
//! Pipeline A (`compile_architecture_with_topology`):
//! - Maps capabilities to `FeatureGroup`s, `NeuronAssignment`s, and specialized `legal_masks`
//! - Classifies parameter generation into `ParameterMode` (Direct / Learned / Hybrid / SACI / Formal)
//! - Extracts inter-feature WSP communication edges (`FeatureDependency`)
//! - Binds `SaciOpId`s (`SaciBinding`) and formal `VerificationObligation`s.

use crate::capability::{
    CapabilityGraph, CapabilityId, CapabilitySpec, CompilationTarget, ObjectiveKind, SemanticsKind,
};
use crate::ir::NeuronAssignment;
use crate::sys::{ErrorCode, FeatureFamily, NeuronId, SaciOpId};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FeatureGroupId(pub u16);

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterMode {
    DirectConstruct = 0,
    LearnedOptimize = 1,
    HybridInitAndTrain = 2,
    ExternalSaci = 3,
    FormalChecker = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureGroup {
    pub group_id: FeatureGroupId,
    pub capability_id: CapabilityId,
    pub family: FeatureFamily,
    pub target: CompilationTarget,
    pub param_mode: ParameterMode,
    pub neuron_start: u32,
    pub neuron_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureParameterSpec {
    pub group_id: FeatureGroupId,
    pub param_mode: ParameterMode,
    pub semantics: SemanticsKind,
    pub objective: ObjectiveKind,
    pub constraint_weight: f64,
    pub target_tolerance: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureDependency {
    pub from_group: FeatureGroupId,
    pub to_group: FeatureGroupId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaciBinding {
    pub capability_id: CapabilityId,
    pub group_id: FeatureGroupId,
    pub op_id: SaciOpId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationObligation {
    pub capability_id: CapabilityId,
    pub preconditions: u32,
    pub postconditions: u32,
    pub invariants: u32,
    pub required_conditions: u32,
    pub tolerance: f64,
    pub requires_formal_proof: bool,
}

#[derive(Clone, Debug)]
pub struct FeatureNeuronManifest {
    pub num_neurons: usize,
    pub feature_groups: Vec<FeatureGroup>,
    pub neuron_assignments: Vec<NeuronAssignment>,
    pub parameter_specs: Vec<FeatureParameterSpec>,
    pub dependencies: Vec<FeatureDependency>,
    pub saci_bindings: Vec<SaciBinding>,
    pub verification_obligations: Vec<VerificationObligation>,
    /// Specialized per-neuron eligibility masks derived from `neuron_assignments`.
    pub legal_masks: Vec<u8>,
}

#[inline]
const fn target_to_parameter_mode(target: CompilationTarget) -> ParameterMode {
    match target {
        CompilationTarget::DirectCircuit => ParameterMode::DirectConstruct,
        CompilationTarget::NeuralWeights | CompilationTarget::LearnedPolicyAndSaci => {
            ParameterMode::LearnedOptimize
        }
        CompilationTarget::SimulationAndSurrogate => ParameterMode::HybridInitAndTrain,
        CompilationTarget::SaciExpert => ParameterMode::ExternalSaci,
        CompilationTarget::FormalCheckerAndDetector => ParameterMode::FormalChecker,
    }
}

/// Lower a validated `CapabilityGraph` + `CapabilitySpec` slice into a
/// `FeatureNeuronManifest` with specialized per-neuron `legal_masks`.
pub fn lower_capabilities(
    graph: &CapabilityGraph,
    specs: &[CapabilitySpec],
    num_neurons: usize,
    neurons_per_group: usize,
) -> Result<FeatureNeuronManifest, ErrorCode> {
    if num_neurons == 0 || neurons_per_group == 0 {
        return Err(ErrorCode::InvalidArchitecture);
    }
    if graph.nodes.len() != specs.len() {
        return Err(ErrorCode::InvalidArchitecture);
    }

    let n = graph.nodes.len();
    let mut feature_groups = Vec::with_capacity(n);
    let mut neuron_assignments = Vec::new();
    let mut parameter_specs = Vec::with_capacity(n);
    let mut saci_bindings = Vec::new();
    let mut verification_obligations = Vec::with_capacity(n);
    let mut legal_masks = vec![0u8; num_neurons];
    let mut node_to_group = vec![FeatureGroupId(0); n];

    let mut next_neuron = 0usize;

    for (g_idx, &node_idx_u16) in graph.topo_order.iter().enumerate() {
        let node_idx = node_idx_u16 as usize;
        let node = &graph.nodes[node_idx];
        let spec = &specs[node_idx];
        let group_id = FeatureGroupId(g_idx as u16);
        node_to_group[node_idx] = group_id;

        let param_mode = target_to_parameter_mode(node.target);

        // Formal verification subsystems do not require dedicated neural weights
        // unless paired with a learned detector; allocate neurons for all neural/interface groups.
        let alloc_count = if node.target == CompilationTarget::FormalCheckerAndDetector {
            0
        } else {
            if next_neuron + neurons_per_group > num_neurons {
                return Err(ErrorCode::CapacityExceeded);
            }
            neurons_per_group
        };

        let neuron_start = next_neuron as u32;
        for offset in 0..alloc_count {
            let nid = next_neuron + offset;
            legal_masks[nid] = node.family.bit();
            neuron_assignments.push(NeuronAssignment {
                neuron: NeuronId(nid as u32),
                family: node.family,
            });
        }
        next_neuron += alloc_count;

        feature_groups.push(FeatureGroup {
            group_id,
            capability_id: node.id,
            family: node.family,
            target: node.target,
            param_mode,
            neuron_start,
            neuron_count: alloc_count as u32,
        });

        let (objective, constraint_weight, target_tolerance) = match spec.training_objective {
            Some(obj) => (obj.kind, obj.constraint_weight, obj.target_tolerance),
            None => (ObjectiveKind::None, 0.0, spec.verification_spec.tolerance),
        };

        parameter_specs.push(FeatureParameterSpec {
            group_id,
            param_mode,
            semantics: spec.semantics,
            objective,
            constraint_weight,
            target_tolerance,
        });

        if let Some(op_id) = spec.saci_binding {
            saci_bindings.push(SaciBinding {
                capability_id: node.id,
                group_id,
                op_id,
            });
        }

        verification_obligations.push(VerificationObligation {
            capability_id: node.id,
            preconditions: spec.preconditions,
            postconditions: spec.postconditions,
            invariants: spec.invariants,
            required_conditions: spec.verification_spec.required_conditions,
            tolerance: spec.verification_spec.tolerance,
            requires_formal_proof: spec.verification_spec.requires_formal_proof,
        });
    }

    // Extract WSP inter-feature communication edges from the CapabilityGraph CSR DAG
    let mut dependencies = Vec::with_capacity(graph.fwd_edges.len());
    for u in 0..n {
        let from_group = node_to_group[u];
        for &v in graph.dependents(u as u16) {
            let to_group = node_to_group[v as usize];
            dependencies.push(FeatureDependency {
                from_group,
                to_group,
            });
        }
    }

    Ok(FeatureNeuronManifest {
        num_neurons,
        feature_groups,
        neuron_assignments,
        parameter_specs,
        dependencies,
        saci_bindings,
        verification_obligations,
        legal_masks,
    })
}
