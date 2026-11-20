//! WSP Compiler (Static + Dynamic) & Sparse Lowering (Priorities 2 & 3).
//!
//! - Static WSP: derives legal pathways and tile metadata from `ArchitectureIr`.
//! - Dynamic WSP: validates query plans, generates per-neuron bitpacked masks,
//!   and formally enforces `sum_f z_{i,f} <= 1` (`ExclusivityViolation` if violated).
//! - Sparse Lowering & Execution: compacts masks into active-neuron and
//!   structured-tile lists and physically skips inactive neurons/tiles.
//!
//! Zero runtime heap allocation on the query hot path.

use crate::etf::FeatureEtfIr;
use crate::ir::{ArchitectureIr, WorkflowPlanIr};
use crate::sys::{
    ErrorCode, EventKind, FeatureFamily, Module, NeuronAuditRecord, NeuronId, QueryId, SaciOpId,
    Status, TelemetryBus, FLAG_FAILED, FLAG_SPARSE_SKIP, FLAG_VALIDATED, MASK_INACTIVE,
    MASK_KNOWLEDGE, MASK_REASONING,
};

#[derive(Clone, Debug)]
pub struct StaticWspMetadata {
    pub num_layers: usize,
    pub num_neurons: usize,
    pub substrate_dim: usize,
    pub tile_size: usize,
    pub num_tiles: usize,
    pub legal_masks: Vec<u8>,
    pub legal_saci_ops: Vec<SaciOpId>,
}

impl StaticWspMetadata {
    pub fn from_arch_ir(ir: &ArchitectureIr) -> Self {
        let num_layers = ir.attention.layers.len();
        let num_neurons = ir.feature_etf.num_neurons;
        let tile_size = ir.tile_size.max(1);
        let num_tiles = (num_neurons + tile_size - 1) / tile_size;
        let legal_saci_ops = ir.saci.ops.iter().map(|op| op.id).collect();
        Self {
            num_layers,
            num_neurons,
            substrate_dim: ir.feature_etf.substrate_dim,
            tile_size,
            num_tiles,
            legal_masks: ir.feature_etf.legal_masks.clone(),
            legal_saci_ops,
        }
    }

    #[inline]
    pub fn is_saci_op_legal(&self, op: SaciOpId) -> bool {
        self.legal_saci_ops.contains(&op)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseStats {
    pub total_neurons: u32,
    pub active_neurons: u32,
    pub skipped_neurons: u32,
    pub total_tiles: u32,
    pub active_tiles: u32,
    pub total_flops: u64,
    pub active_flops: u64,
    pub skipped_flops: u64,
}

/// Preallocated scratchpad for Dynamic WSP mask generation, sparse lowering,
/// and sparse execution (zero runtime heap allocations).
pub struct WspScratch {
    pub num_neurons: usize,
    pub num_tiles: usize,
    pub neuron_masks: Vec<u8>,
    /// Multi-layer activation matrix `M_q in {0,1}^{L x N x 4}` (`num_layers * num_neurons`)
    pub layer_masks: Vec<u8>,
    /// Computational capability mask `M_compute` over dynamic SACI ops
    pub compute_mask: Vec<u64>,
    pub active_neurons: Vec<NeuronId>,
    pub active_families: Vec<FeatureFamily>,
    pub active_count: usize,
    pub tile_active: Vec<u8>,
    pub active_tile_ids: Vec<u32>,
    pub active_tiles_count: usize,
    pub per_family_active: [u32; 4],
    pub per_family_suppressed: [u32; 4],
}

impl WspScratch {
    pub fn for_static(wsp: &StaticWspMetadata) -> Self {
        let compute_words = wsp.legal_saci_ops.len().max(256).div_ceil(64);
        Self {
            num_neurons: wsp.num_neurons,
            num_tiles: wsp.num_tiles,
            neuron_masks: vec![MASK_INACTIVE; wsp.num_neurons],
            layer_masks: vec![MASK_INACTIVE; wsp.num_layers * wsp.num_neurons],
            compute_mask: vec![0; compute_words],
            active_neurons: vec![NeuronId(0); wsp.num_neurons],
            active_families: vec![FeatureFamily::Knowledge; wsp.num_neurons],
            active_count: 0,
            tile_active: vec![0; wsp.num_tiles],
            active_tile_ids: vec![0; wsp.num_tiles],
            active_tiles_count: 0,
            per_family_active: [0; 4],
            per_family_suppressed: [0; 4],
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.neuron_masks.fill(MASK_INACTIVE);
        self.layer_masks.fill(MASK_INACTIVE);
        self.compute_mask.fill(0);
        self.tile_active.fill(0);
        self.active_count = 0;
        self.active_tiles_count = 0;
        self.per_family_active = [0; 4];
        self.per_family_suppressed = [0; 4];
    }

    #[inline]
    pub fn set_compute_op(&mut self, op_idx: usize) {
        let word = op_idx >> 6;
        if word < self.compute_mask.len() {
            self.compute_mask[word] |= 1u64 << (op_idx & 63);
        }
    }

    #[inline]
    pub fn is_compute_op_active(&self, op_idx: usize) -> bool {
        let word = op_idx >> 6;
        word < self.compute_mask.len() && (self.compute_mask[word] & (1u64 << (op_idx & 63))) != 0
    }
}

/// Verify `sum_f z_{i,f} <= 1` for every neuron mask in `O(N)`.
#[inline]
pub fn verify_exclusivity(masks: &[u8]) -> Result<(), ErrorCode> {
    for &m in masks {
        if m.count_ones() > 1 {
            return Err(ErrorCode::ExclusivityViolation);
        }
    }
    Ok(())
}

/// Dynamic WSP: validate query plan, generate per-neuron masks, and enforce
/// `sum_f z_{i,f} <= 1` + pathway legality.
pub fn generate_wsp_masks<const CAP: usize>(
    static_wsp: &StaticWspMetadata,
    plan: &WorkflowPlanIr<'_>,
    scratch: &mut WspScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(), ErrorCode> {
    if scratch.num_neurons < static_wsp.num_neurons {
        return Err(ErrorCode::BufferTooSmall);
    }
    scratch.reset();

    // 1. Validate SACI capabilities requested by plan
    for &op_id in plan.saci_ops {
        if !static_wsp.is_saci_op_legal(op_id) {
            telemetry.emit(
                Module::WspDynamic,
                EventKind::Error,
                Status::Failed,
                ErrorCode::UnknownSaciOp,
                FLAG_FAILED,
                op_id.0 as u32,
                0,
            );
            return Err(ErrorCode::UnknownSaciOp);
        }
    }

    // 2. Assign per-neuron masks while enforcing legality and exclusivity
    for &assign in plan.assignments {
        let idx = assign.neuron.0 as usize;
        if idx >= static_wsp.num_neurons {
            return Err(ErrorCode::IllegalPathway);
        }
        let bit = assign.family.bit();
        if (static_wsp.legal_masks[idx] & bit) == 0 {
            telemetry.emit(
                Module::WspDynamic,
                EventKind::Error,
                Status::Failed,
                ErrorCode::IllegalPathway,
                FLAG_FAILED,
                assign.neuron.0,
                assign.family as u32,
            );
            return Err(ErrorCode::IllegalPathway);
        }

        let new_mask = scratch.neuron_masks[idx] | bit;
        // Enforce sum_{f} z_{i,f} <= 1 (and reject duplicate conflicting/redundant activation)
        if new_mask.count_ones() > 1 || scratch.neuron_masks[idx] != MASK_INACTIVE {
            telemetry.emit(
                Module::WspDynamic,
                EventKind::ExclusivityChecked,
                Status::Failed,
                ErrorCode::ExclusivityViolation,
                FLAG_FAILED,
                assign.neuron.0,
                new_mask as u32,
            );
            return Err(ErrorCode::ExclusivityViolation);
        }
        scratch.neuron_masks[idx] = new_mask;
    }

    verify_exclusivity(&scratch.neuron_masks)?;
    // Populate M_compute for requested SACI ops
    for &op_id in plan.saci_ops {
        if let Some(pos) = static_wsp.legal_saci_ops.iter().position(|&id| id == op_id) {
            scratch.set_compute_op(pos);
        }
    }
    telemetry.emit(
        Module::WspDynamic,
        EventKind::WspMasksGenerated,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        plan.assignments.len() as u32,
        static_wsp.num_neurons as u32,
    );
    Ok(())
}

/// Phase 1: Bootstrap Routing (Stage A1).
/// Activates ONLY a bounded planning subset of Knowledge & Reasoning neurons;
/// SACI and Simulation remain strictly inactive during bootstrap.
pub fn bootstrap_wsp<const CAP: usize>(
    static_wsp: &StaticWspMetadata,
    vq_code: u32,
    bootstrap_budget: usize,
    scratch: &mut WspScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<usize, ErrorCode> {
    scratch.reset();
    let n = static_wsp.num_neurons;
    let budget = bootstrap_budget.min(n);
    let mut activated = 0usize;

    for step in 0..budget {
        // Deterministic spread across substrate based on VQ codebook ID
        let nid = ((vq_code as usize).wrapping_mul(7) + step) % n;
        if scratch.neuron_masks[nid] != MASK_INACTIVE {
            continue;
        }
        let legal = static_wsp.legal_masks[nid];
        let fam_mask = if step % 2 == 0 && (legal & MASK_KNOWLEDGE) != 0 {
            MASK_KNOWLEDGE
        } else if (legal & MASK_REASONING) != 0 {
            MASK_REASONING
        } else if (legal & MASK_KNOWLEDGE) != 0 {
            MASK_KNOWLEDGE
        } else {
            continue;
        };
        scratch.neuron_masks[nid] = fam_mask;
        activated += 1;
    }

    verify_exclusivity(&scratch.neuron_masks)?;
    telemetry.emit(
        Module::WspDynamic,
        EventKind::BootstrapRouted,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        vq_code,
        activated as u32,
    );
    Ok(activated)
}

/// Stage B: Score-based hard feature selection for layer `layer_id` (`0..8`):
///   z_{i,f} = 1 iff f = argmax_j s_{i,j} and s_{i,f} > tau (and legal).
/// Enforces sum_f z_{i,f} <= 1 and optionally records `NeuronAuditRecord`s.
pub fn score_and_select_features<const CAP: usize>(
    static_wsp: &StaticWspMetadata,
    query_id: QueryId,
    revision: u32,
    layer_id: u8,
    scores: &[[f32; 4]],
    tau: f32,
    scratch: &mut WspScratch,
    audit_out: Option<&mut Vec<NeuronAuditRecord>>,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<usize, ErrorCode> {
    let n = static_wsp.num_neurons;
    if scores.len() != n || (layer_id as usize) >= static_wsp.num_layers {
        return Err(ErrorCode::BufferTooSmall);
    }

    let layer_off = (layer_id as usize) * n;
    let mut active_in_layer = 0usize;

    let mut audit_vec = audit_out;
    for i in 0..n {
        let legal = static_wsp.legal_masks[i];
        let mut best_f: Option<usize> = None;
        let mut best_s = tau;

        for f in 0..4 {
            let bit = 1u8 << f;
            if (legal & bit) != 0 && scores[i][f] > best_s {
                best_s = scores[i][f];
                best_f = Some(f);
            }
        }

        let mask = match best_f {
            Some(f) => {
                scratch.per_family_active[f] += 1;
                for other in 0..4 {
                    if other != f {
                        scratch.per_family_suppressed[other] += 1;
                    }
                }
                active_in_layer += 1;
                1u8 << f
            }
            None => {
                for f in 0..4 {
                    scratch.per_family_suppressed[f] += 1;
                }
                MASK_INACTIVE
            }
        };

        scratch.layer_masks[layer_off + i] = mask;
        scratch.neuron_masks[i] = mask;

        if let Some(ref mut log) = audit_vec {
            log.push(NeuronAuditRecord {
                query_id,
                revision,
                layer: layer_id,
                selected_mask: mask,
                neuron_id: i as u32,
                scores: scores[i],
            });
        }
    }

    verify_exclusivity(&scratch.layer_masks[layer_off..layer_off + n])?;
    telemetry.emit(
        Module::WspDynamic,
        EventKind::ExclusivityChecked,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        layer_id as u32,
        active_in_layer as u32,
    );
    Ok(active_in_layer)
}

/// Lower bitpacked WSP masks into compact active-neuron and structured-tile schedules.
pub fn lower_sparse_schedule<const CAP: usize>(
    static_wsp: &StaticWspMetadata,
    scratch: &mut WspScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> SparseStats {
    let mut active_count = 0usize;
    let mut active_tiles_count = 0usize;
    let tile_sz = static_wsp.tile_size;

    for (idx, &mask) in scratch.neuron_masks[..static_wsp.num_neurons]
        .iter()
        .enumerate()
    {
        if mask == MASK_INACTIVE {
            continue;
        }
        let family = match mask {
            1 => FeatureFamily::Knowledge,
            2 => FeatureFamily::Reasoning,
            4 => FeatureFamily::SaciInterface,
            8 => FeatureFamily::Simulation,
            _ => continue,
        };
        scratch.active_neurons[active_count] = NeuronId(idx as u32);
        scratch.active_families[active_count] = family;
        active_count += 1;

        let tile_id = idx / tile_sz;
        if scratch.tile_active[tile_id] == 0 {
            scratch.tile_active[tile_id] = 1;
            scratch.active_tile_ids[active_tiles_count] = tile_id as u32;
            active_tiles_count += 1;
        }
    }

    scratch.active_count = active_count;
    scratch.active_tiles_count = active_tiles_count;

    // 2 * substrate_dim MACs (read projection + write projection) per active neuron
    let flops_per_neuron = (2 * static_wsp.substrate_dim) as u64;
    let total_neurons = static_wsp.num_neurons as u32;
    let active_u32 = active_count as u32;
    let skipped_u32 = total_neurons - active_u32;

    let stats = SparseStats {
        total_neurons,
        active_neurons: active_u32,
        skipped_neurons: skipped_u32,
        total_tiles: static_wsp.num_tiles as u32,
        active_tiles: active_tiles_count as u32,
        total_flops: (total_neurons as u64) * flops_per_neuron,
        active_flops: (active_u32 as u64) * flops_per_neuron,
        skipped_flops: (skipped_u32 as u64) * flops_per_neuron,
    };

    telemetry.emit(
        Module::SparseLowering,
        EventKind::SparseScheduleLowered,
        Status::Completed,
        ErrorCode::None,
        FLAG_SPARSE_SKIP,
        active_u32,
        skipped_u32,
    );

    stats
}

/// Execute ONLY the active neurons in `scratch`, physically skipping all inactive neurons.
pub fn execute_sparse_substrate<const CAP: usize>(
    etf: &FeatureEtfIr,
    scratch: &WspScratch,
    input: &[f64],
    output: &mut [f64],
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<SparseStats, ErrorCode> {
    let dim = etf.substrate_dim;
    if input.len() != dim || output.len() != dim {
        return Err(ErrorCode::BufferTooSmall);
    }

    output.fill(0.0);
    let mut executed_flops = 0u64;

    for k in 0..scratch.active_count {
        let nid = scratch.active_neurons[k].0 as usize;
        let fam = scratch.active_families[k];
        let fvec = etf.feature_vec(nid, fam);

        // Read projection: dot(input, f_{nid, fam})
        let mut act = 0.0_f64;
        for d in 0..dim {
            act += input[d] * fvec[d];
        }
        // ReLU activation
        if act > 0.0 {
            for d in 0..dim {
                output[d] += act * fvec[d];
            }
        }
        executed_flops += (2 * dim) as u64;
    }

    let total_neurons = etf.num_neurons as u32;
    let active_u32 = scratch.active_count as u32;
    let skipped_u32 = total_neurons - active_u32;
    let total_flops = (total_neurons as u64) * ((2 * dim) as u64);

    telemetry.emit(
        Module::Runtime,
        EventKind::SparseDispatchExecuted,
        Status::Completed,
        ErrorCode::None,
        FLAG_SPARSE_SKIP,
        active_u32,
        skipped_u32,
    );

    Ok(SparseStats {
        total_neurons,
        active_neurons: active_u32,
        skipped_neurons: skipped_u32,
        total_tiles: scratch.num_tiles as u32,
        active_tiles: scratch.active_tiles_count as u32,
        total_flops,
        active_flops: executed_flops,
        skipped_flops: total_flops - executed_flops,
    })
}
