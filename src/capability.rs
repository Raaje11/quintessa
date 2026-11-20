//! Capability Graph & Capability Classification Pass.
//!
//! Separates *what* a domain capability computes (formal semantics, typed I/O,
//! preconditions, postconditions, invariants, and verification criteria) from
//! *how* it is realized (`CompilationTarget`: neural weights, direct circuit,
//! SACI expert, simulation + surrogate, or formal checker) and *how* capabilities
//! communicate (WSP dependency edges).

use crate::sys::{ArrayVec, ErrorCode, FeatureFamily, SaciOpId};

pub const MAX_CAP_IO: usize = 8;
pub const MAX_CAP_DEPS: usize = 8;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityId(pub u16);

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CapabilityType {
    Knowledge = 0,
    Reasoning = 1,
    Simulation = 2,
    Verification = 3,
    Computation = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompilationTarget {
    NeuralWeights = 0,
    LearnedPolicyAndSaci = 1,
    DirectCircuit = 2,
    SimulationAndSurrogate = 3,
    FormalCheckerAndDetector = 4,
    SaciExpert = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    Float = 0,
    Int = 1,
    Bool = 2,
    Vector = 3,
    Matrix = 4,
    State = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticsKind {
    ConceptRelation = 0,
    InferencePolicy = 1,
    StateTransition = 2,
    InvariantPredicate = 3,
    DeterministicFormula = 4,
    FiniteLookup = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveKind {
    None = 0,
    ContrastiveSemantic = 1,
    TracePolicyImitation = 2,
    PhysicsConstrainedTrajectory = 3,
    CounterexampleClassification = 4,
    ResidualRegression = 5,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingObjective {
    pub kind: ObjectiveKind,
    pub constraint_weight: f64,
    pub target_tolerance: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationSpec {
    pub required_conditions: u32,
    pub tolerance: f64,
    pub requires_formal_proof: bool,
}

#[derive(Clone, Debug)]
pub struct CapabilitySpec {
    pub id: CapabilityId,
    pub cap_type: CapabilityType,
    pub inputs: ArrayVec<ValueType, MAX_CAP_IO>,
    pub outputs: ArrayVec<ValueType, MAX_CAP_IO>,
    pub semantics: SemanticsKind,
    pub preconditions: u32,
    pub postconditions: u32,
    pub invariants: u32,
    pub dependencies: ArrayVec<CapabilityId, MAX_CAP_DEPS>,
    pub training_objective: Option<TrainingObjective>,
    pub verification_spec: VerificationSpec,
    pub saci_binding: Option<SaciOpId>,
}

/// Formal capability classification pass:
/// Determines whether a capability becomes learned neural weights, a directly
/// compiled circuit, a SACI expert, a simulation+surrogate, or a formal checker.
pub fn classify_capability(spec: &CapabilitySpec) -> Result<CompilationTarget, ErrorCode> {
    if !spec.verification_spec.tolerance.is_finite() || spec.verification_spec.tolerance < 0.0 {
        return Err(ErrorCode::InvalidArchitecture);
    }

    match spec.cap_type {
        CapabilityType::Knowledge => {
            if spec.semantics == SemanticsKind::FiniteLookup && spec.training_objective.is_none() {
                Ok(CompilationTarget::DirectCircuit)
            } else if spec.training_objective.is_some() {
                Ok(CompilationTarget::NeuralWeights)
            } else {
                // Knowledge weights require an explicit training objective
                Err(ErrorCode::InvalidArchitecture)
            }
        }
        CapabilityType::Reasoning => {
            if spec.training_objective.is_none() {
                return Err(ErrorCode::InvalidArchitecture);
            }
            Ok(CompilationTarget::LearnedPolicyAndSaci)
        }
        CapabilityType::Simulation => {
            if spec.semantics != SemanticsKind::StateTransition {
                return Err(ErrorCode::InvalidArchitecture);
            }
            Ok(CompilationTarget::SimulationAndSurrogate)
        }
        CapabilityType::Verification => {
            if !spec.verification_spec.requires_formal_proof
                || (spec.verification_spec.required_conditions == 0
                    && spec.invariants == 0
                    && spec.postconditions == 0)
            {
                return Err(ErrorCode::InvalidArchitecture);
            }
            Ok(CompilationTarget::FormalCheckerAndDetector)
        }
        CapabilityType::Computation => {
            if spec.semantics == SemanticsKind::FiniteLookup && spec.saci_binding.is_none() {
                Ok(CompilationTarget::DirectCircuit)
            } else if spec.saci_binding.is_some() {
                Ok(CompilationTarget::SaciExpert)
            } else {
                Err(ErrorCode::UnknownSaciOp)
            }
        }
    }
}

/// Map a capability type to its primary ADA FeatureFamily slot.
#[inline]
pub const fn capability_feature_family(cap_type: CapabilityType) -> FeatureFamily {
    match cap_type {
        CapabilityType::Knowledge => FeatureFamily::Knowledge,
        CapabilityType::Reasoning | CapabilityType::Verification => FeatureFamily::Reasoning,
        CapabilityType::Computation => FeatureFamily::SaciInterface,
        CapabilityType::Simulation => FeatureFamily::Simulation,
    }
}

/// Formal verifier `V(x) in {0, 1}`: checks preconditions, postconditions,
/// invariants, and numerical tolerance without relying on neural confidence.
#[inline]
pub fn verify_capability_contract(
    spec: &CapabilitySpec,
    observed_pre: u32,
    observed_post: u32,
    observed_inv: u32,
    observed_error: f64,
) -> bool {
    let pre_ok = (observed_pre & spec.preconditions) == spec.preconditions;
    let post_ok = (observed_post & spec.postconditions) == spec.postconditions;
    let inv_ok = (observed_inv & spec.invariants) == spec.invariants;
    let cond_ok = (observed_post & spec.verification_spec.required_conditions)
        == spec.verification_spec.required_conditions;
    let err_ok = observed_error.is_finite() && observed_error <= spec.verification_spec.tolerance;
    pre_ok && post_ok && inv_ok && cond_ok && err_ok
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassifiedCapability {
    pub id: CapabilityId,
    pub cap_type: CapabilityType,
    pub target: CompilationTarget,
    pub family: FeatureFamily,
    pub saci_binding: Option<SaciOpId>,
}

/// Compiled Capability Graph in CSR format (`prereq -> dependent`), with
/// classified compilation targets and topological execution order.
#[derive(Clone, Debug)]
pub struct CapabilityGraph {
    pub nodes: Vec<ClassifiedCapability>,
    pub fwd_offsets: Vec<u32>,
    pub fwd_edges: Vec<u16>,
    pub in_degrees: Vec<u16>,
    pub topo_order: Vec<u16>,
    pub target_counts: [u16; 6],
}

impl CapabilityGraph {
    pub fn compile(specs: &[CapabilitySpec]) -> Result<Self, ErrorCode> {
        let n = specs.len();
        if n == 0 {
            return Err(ErrorCode::InvalidArchitecture);
        }

        let mut nodes = Vec::with_capacity(n);
        let mut target_counts = [0u16; 6];
        for (i, s) in specs.iter().enumerate() {
            // Reject duplicate IDs
            if specs[..i].iter().any(|prev| prev.id == s.id) {
                return Err(ErrorCode::InvalidArchitecture);
            }
            let target = classify_capability(s)?;
            target_counts[target as usize] += 1;
            nodes.push(ClassifiedCapability {
                id: s.id,
                cap_type: s.cap_type,
                target,
                family: capability_feature_family(s.cap_type),
                saci_binding: s.saci_binding,
            });
        }

        // Resolve dependency edges (dep -> dependent)
        let mut counts = vec![0u32; n];
        let mut in_degrees = vec![0u16; n];
        let mut raw_edges: Vec<(u16, u16)> = Vec::new();

        for (dep_idx, s) in specs.iter().enumerate() {
            for &req_id in s.dependencies.as_slice() {
                let req_idx = specs
                    .iter()
                    .position(|other| other.id == req_id)
                    .ok_or(ErrorCode::IllegalPathway)?;
                if req_idx == dep_idx {
                    return Err(ErrorCode::InvalidArchitecture);
                }
                counts[req_idx] += 1;
                in_degrees[dep_idx] += 1;
                raw_edges.push((req_idx as u16, dep_idx as u16));
            }
        }

        let mut fwd_offsets = vec![0u32; n + 1];
        for i in 0..n {
            fwd_offsets[i + 1] = fwd_offsets[i] + counts[i];
        }

        let mut fwd_edges = vec![0u16; raw_edges.len()];
        let mut cursor = fwd_offsets[..n].to_vec();
        for &(u, v) in &raw_edges {
            let idx = cursor[u as usize] as usize;
            fwd_edges[idx] = v;
            cursor[u as usize] += 1;
        }

        // Kahn's topological sort to validate acyclicity
        let mut tmp_deg = in_degrees.clone();
        let mut queue = Vec::with_capacity(n);
        for (i, &d) in tmp_deg.iter().enumerate() {
            if d == 0 {
                queue.push(i as u16);
            }
        }

        let mut topo_order = Vec::with_capacity(n);
        while let Some(u) = queue.pop() {
            topo_order.push(u);
            let s = fwd_offsets[u as usize] as usize;
            let e = fwd_offsets[u as usize + 1] as usize;
            for &v in &fwd_edges[s..e] {
                let d = &mut tmp_deg[v as usize];
                *d -= 1;
                if *d == 0 {
                    queue.push(v);
                }
            }
        }

        if topo_order.len() != n {
            return Err(ErrorCode::NcmCycleDetected);
        }

        Ok(Self {
            nodes,
            fwd_offsets,
            fwd_edges,
            in_degrees,
            topo_order,
            target_counts,
        })
    }

    #[inline]
    pub fn dependents(&self, idx: u16) -> &[u16] {
        let i = idx as usize;
        let s = self.fwd_offsets[i] as usize;
        let e = self.fwd_offsets[i + 1] as usize;
        &self.fwd_edges[s..e]
    }
}
