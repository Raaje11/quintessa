//! Unified ADA-IR: Authoritative Single Source of Truth.
//!
//! All eight IR views (Architecture, Feature/ETF, Attention, Workflow/Plan,
//! SACI, Workspace, NCM Output, Telemetry) derive their structural metadata
//! from this validated representation.

use crate::etf::FeatureEtfIr;
use crate::sys::{
    ArrayVec, AttentionKind, Backend, ErrorCode, FeatureFamily, LayerId, NcmNodeId, NeuronId,
    NumericalMode, SaciOpId,
};

pub const ADA_NUM_ATTENTION_LAYERS: usize = 8;

/// Exact 6 KDA + 2 MLA hybrid topology:
/// `[KDA, KDA, KDA, MLA, KDA, KDA, KDA, MLA]`
pub const ADA_ATTENTION_TOPOLOGY: [AttentionKind; ADA_NUM_ATTENTION_LAYERS] = [
    AttentionKind::Kda,
    AttentionKind::Kda,
    AttentionKind::Kda,
    AttentionKind::Mla,
    AttentionKind::Kda,
    AttentionKind::Kda,
    AttentionKind::Kda,
    AttentionKind::Mla,
];

#[derive(Clone, Copy, Debug)]
pub struct KdaConfig {
    pub chunk_size: u16,
    pub dplr_rank: u16,
    pub state_dim: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct MlaConfig {
    pub kv_latent_dim: u16,
    pub num_heads: u16,
    pub head_dim: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct LayerSpec {
    pub id: LayerId,
    pub kind: AttentionKind,
    pub kda: KdaConfig,
    pub mla: MlaConfig,
}

#[derive(Clone, Debug)]
pub struct AttentionIr {
    pub layers: Vec<LayerSpec>,
}

impl AttentionIr {
    pub fn build(
        topology: &[AttentionKind],
        kda: KdaConfig,
        mla: MlaConfig,
    ) -> Result<Self, ErrorCode> {
        if topology.is_empty() || topology.len() > 256 {
            return Err(ErrorCode::InvalidAttentionTopology);
        }
        let mut layers = Vec::with_capacity(topology.len());
        for (i, &kind) in topology.iter().enumerate() {
            layers.push(LayerSpec {
                id: LayerId(i as u8),
                kind,
                kda,
                mla,
            });
        }
        Ok(Self { layers })
    }

    pub fn from_layers(layers: Vec<LayerSpec>) -> Result<Self, ErrorCode> {
        if layers.is_empty() || layers.len() > 256 {
            return Err(ErrorCode::InvalidAttentionTopology);
        }
        Ok(Self { layers })
    }

    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// SACI IR: Formal Domain Algorithm Contracts (Section 5)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SaciOpSpec {
    pub id: SaciOpId,
    pub numerical_mode: NumericalMode,
    pub tolerance: f64,
    pub backend: Backend,
    pub precondition_flags: u32,
    pub postcondition_flags: u32,
}

#[derive(Clone, Debug)]
pub struct SaciIr {
    pub ops: Vec<SaciOpSpec>,
}

impl SaciIr {
    pub fn new(ops: Vec<SaciOpSpec>) -> Self {
        Self { ops }
    }

    #[inline]
    pub fn lookup(&self, id: SaciOpId) -> Option<&SaciOpSpec> {
        self.ops.iter().find(|op| op.id == id)
    }
}

/// CSR DAG of SACI operations for a query (`(prereq_idx, dependent_idx)`).
#[derive(Clone, Debug)]
pub struct SaciDag {
    pub ops: Vec<SaciOpId>,
    pub fwd_offsets: Vec<u32>,
    pub fwd_edges: Vec<u16>,
    pub in_degrees: Vec<u16>,
}

impl SaciDag {
    pub fn from_ops_and_edges(ops: Vec<SaciOpId>, edges: &[(u16, u16)]) -> Result<Self, ErrorCode> {
        let n = ops.len();
        let mut counts = vec![0u32; n];
        let mut in_degrees = vec![0u16; n];
        for &(u, v) in edges {
            if (u as usize) >= n || (v as usize) >= n || u == v {
                return Err(ErrorCode::InvalidArchitecture);
            }
            counts[u as usize] += 1;
            in_degrees[v as usize] += 1;
        }
        let mut fwd_offsets = vec![0u32; n + 1];
        for i in 0..n {
            fwd_offsets[i + 1] = fwd_offsets[i] + counts[i];
        }
        let mut fwd_edges = vec![0u16; edges.len()];
        let mut cursor = fwd_offsets[..n].to_vec();
        for &(u, v) in edges {
            let idx = cursor[u as usize] as usize;
            fwd_edges[idx] = v;
            cursor[u as usize] += 1;
        }

        // Validate acyclicity
        let mut tmp_deg = in_degrees.clone();
        let mut queue: ArrayVec<u16, 256> = ArrayVec::new();
        for (i, &d) in tmp_deg.iter().enumerate() {
            if d == 0 && queue.push(i as u16) != ErrorCode::None {
                return Err(ErrorCode::CapacityExceeded);
            }
        }
        let mut visited = 0usize;
        while let Some(u) = queue.pop() {
            visited += 1;
            let s = fwd_offsets[u as usize] as usize;
            let e = fwd_offsets[u as usize + 1] as usize;
            for &v in &fwd_edges[s..e] {
                let d = &mut tmp_deg[v as usize];
                *d -= 1;
                if *d == 0 && queue.push(v) != ErrorCode::None {
                    return Err(ErrorCode::CapacityExceeded);
                }
            }
        }
        if visited != n {
            return Err(ErrorCode::NcmCycleDetected);
        }

        Ok(Self {
            ops,
            fwd_offsets,
            fwd_edges,
            in_degrees,
        })
    }

    #[inline]
    pub fn dependents(&self, u: u16) -> &[u16] {
        let i = u as usize;
        let s = self.fwd_offsets[i] as usize;
        let e = self.fwd_offsets[i + 1] as usize;
        &self.fwd_edges[s..e]
    }
}

// ---------------------------------------------------------------------------
// Workspace IR (Section 7)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct WorkspaceIr {
    pub scratchpad_slots: u16,
    pub max_iters: u16,
    pub halting_tolerance: f64,
}

impl Default for WorkspaceIr {
    fn default() -> Self {
        Self {
            scratchpad_slots: 16,
            max_iters: 8,
            halting_tolerance: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// NCM Output IR: CSR Dependency Graph (Section 8)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcmArtifactKind {
    Title = 0,
    GeometrySpec = 1,
    MathResults = 2,
    TechnicalExplanation = 3,
    Tables = 4,
    ValidationSummary = 5,
    DomainArtifact = 6,
    CustomPayload = 7,
}

/// CSR DAG of NCM output components (`dep -> dependent`).
#[derive(Clone, Debug)]
pub struct NcmOutputIr {
    pub num_nodes: usize,
    pub kinds: Vec<NcmArtifactKind>,
    pub fwd_offsets: Vec<u32>,
    pub fwd_edges: Vec<NcmNodeId>,
    pub in_degrees: Vec<u16>,
}

impl NcmOutputIr {
    /// Build and validate an acyclic CSR NCM output graph from `(prereq, dependent)` pairs.
    pub fn from_edges(
        kinds: Vec<NcmArtifactKind>,
        edges: &[(NcmNodeId, NcmNodeId)],
    ) -> Result<Self, ErrorCode> {
        let num_nodes = kinds.len();
        if num_nodes == 0 {
            return Err(ErrorCode::InvalidNcmGraph);
        }

        let mut counts = vec![0u32; num_nodes];
        let mut in_degrees = vec![0u16; num_nodes];
        for &(u, v) in edges {
            if (u.0 as usize) >= num_nodes || (v.0 as usize) >= num_nodes || u == v {
                return Err(ErrorCode::InvalidNcmGraph);
            }
            counts[u.0 as usize] += 1;
            in_degrees[v.0 as usize] += 1;
        }

        let mut fwd_offsets = vec![0u32; num_nodes + 1];
        for i in 0..num_nodes {
            fwd_offsets[i + 1] = fwd_offsets[i] + counts[i];
        }

        let mut fwd_edges = vec![NcmNodeId(0); edges.len()];
        let mut cursor = fwd_offsets[..num_nodes].to_vec();
        for &(u, v) in edges {
            let idx = cursor[u.0 as usize] as usize;
            fwd_edges[idx] = v;
            cursor[u.0 as usize] += 1;
        }

        let ir = Self {
            num_nodes,
            kinds,
            fwd_offsets,
            fwd_edges,
            in_degrees,
        };

        // Validate acyclicity using Kahn's algorithm
        let mut tmp_deg = ir.in_degrees.clone();
        let mut queue = Vec::with_capacity(num_nodes);
        for (i, &deg) in tmp_deg.iter().enumerate() {
            if deg == 0 {
                queue.push(NcmNodeId(i as u16));
            }
        }
        let mut visited = 0usize;
        while let Some(u) = queue.pop() {
            visited += 1;
            for &v in ir.dependents(u) {
                let d = &mut tmp_deg[v.0 as usize];
                *d -= 1;
                if *d == 0 {
                    queue.push(v);
                }
            }
        }
        if visited != num_nodes {
            return Err(ErrorCode::NcmCycleDetected);
        }

        Ok(ir)
    }

    #[inline]
    pub fn dependents(&self, u: NcmNodeId) -> &[NcmNodeId] {
        let i = u.0 as usize;
        let s = self.fwd_offsets[i] as usize;
        let e = self.fwd_offsets[i + 1] as usize;
        &self.fwd_edges[s..e]
    }
}

// ---------------------------------------------------------------------------
// Architecture IR (Top-Level Authoritative Model Spec)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct ArchitectureSpec {
    pub vq_codebook_size: u32,
    pub num_neurons: usize,
    pub substrate_dim: usize,
    pub tile_size: usize,
    pub kda: KdaConfig,
    pub mla: MlaConfig,
    pub workspace: WorkspaceIr,
}

#[derive(Clone, Debug)]
pub struct ArchitectureIr {
    pub vq_codebook_size: u32,
    pub tile_size: usize,
    pub feature_etf: FeatureEtfIr,
    pub attention: AttentionIr,
    pub saci: SaciIr,
    pub workspace: WorkspaceIr,
}

impl ArchitectureIr {
    /// Total parameter count across the ETF feature basis, VQ codebook, and attention layers (`u64` safe for 1T+ models).
    pub fn parameter_count_u64(&self) -> u64 {
        let dim = self.feature_etf.substrate_dim as u64;
        let etf_params = self.feature_etf.basis.len() as u64;
        let vq_params = (self.vq_codebook_size as u64).saturating_mul(dim);
        let mut attn_params = 0u64;
        for layer in &self.attention.layers {
            match layer.kind {
                AttentionKind::Kda => {
                    let sd = layer.kda.state_dim as u64;
                    let rank = layer.kda.dplr_rank as u64;
                    let term = 3u64
                        .saturating_mul(dim)
                        .saturating_mul(sd)
                        .saturating_add(2u64.saturating_mul(sd).saturating_mul(rank));
                    attn_params = attn_params.saturating_add(term);
                }
                AttentionKind::Mla => {
                    let ld = layer.mla.kv_latent_dim as u64;
                    let heads = layer.mla.num_heads as u64;
                    let hd = layer.mla.head_dim as u64;
                    let h_hd = heads.saturating_mul(hd);
                    let term = dim
                        .saturating_mul(ld)
                        .saturating_add(2u64.saturating_mul(ld).saturating_mul(h_hd))
                        .saturating_add(2u64.saturating_mul(dim).saturating_mul(h_hd));
                    attn_params = attn_params.saturating_add(term);
                }
            }
        }
        etf_params
            .saturating_add(vq_params)
            .saturating_add(attn_params)
    }

    /// Total parameter count across the ETF feature basis, VQ codebook, and attention layers.
    pub fn parameter_count(&self) -> usize {
        self.parameter_count_u64() as usize
    }
}

// ---------------------------------------------------------------------------
// Workflow / Plan IR (Per-Query Input to Pipeline B)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct NeuronAssignment {
    pub neuron: NeuronId,
    pub family: FeatureFamily,
}

#[derive(Clone, Copy, Debug)]
pub struct WorkflowPlanIr<'a> {
    pub assignments: &'a [NeuronAssignment],
    pub saci_ops: &'a [SaciOpId],
    pub verify_threshold: f64,
}
