//! Dedicated ADA-Specific Pure-Weight Compilation Mechanisms (`ada_compiler.rs`).
//!
//! Kept cleanly separated from Quintessa's general architecture compiler (`sharded_compiler.rs`,
//! `workflow_arch.rs`) so Quintessa remains architecture-agnostic while supporting ADA's
//! 4-family neuron model (`Knowledge`, `Reasoning`, `SaciInterface`, `Simulation`):
//!
//! 1. **Knowledge (`K`)**: Concept/domain/aesthetic representations.
//! 2. **Reasoning (`R`)**: Multi-step planning, parameter selection, verification, and halting.
//! 3. **SACI (`S`)**: Pure-weight computational engines (e.g., photo-editing color matrices,
//!    convolutions, SwiGLU blend modes, piecewise-linear curves) compiled directly into standard
//!    GPU-safe `F32`/`BF16` neural layers — zero external tool calls or sidecar runtimes.
//! 4. **Simulation (`M`)**: Internal latent canvas projections (`write_proj`, `read_proj`,
//!    `verify_proj`) so `Reasoning` can visually/mathematically inspect the emulated output
//!    internally before halting.
//!
//! All emitted artifacts use standard Burn (Rust) `.safetensors` format (`{"format":"burn"}`),
//! standard layer names, GPU-safe `F32`/`BF16` dtypes (no `F64`), and zero compiler fingerprints.

use crate::safetensors_writer::{WeightDtype, WeightOrigin};
use crate::sharded_compiler::{
    plan_scale_model, ScaleLayerKind, ScaleModelSpec, ShardedCompilationPlan,
    DEFAULT_MAX_SHARD_BYTES,
};
use crate::sys::{ErrorCode, FeatureFamily, TelemetryBus};
use crate::workflow_arch::CompileProfile;

// ---------------------------------------------------------------------------
// 0. ADA-1B Eight General Intelligence Domains + Shared Transferable SACI Primitives
// ---------------------------------------------------------------------------

/// The 8 independent General Intelligence domains plus ADA's specialist Parametric CAD domain.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdaCapabilityDomain {
    Semantic = 0,
    Reasoning = 1,
    KnowledgeExplanation = 2,
    Mathematics = 3,
    Programming = 4,
    OfficeBusiness = 5,
    Research = 6,
    ToolUse = 7,
    ParametricCad = 8,
}

impl AdaCapabilityDomain {
    pub const ALL_GENERAL: [Self; 8] = [
        Self::Semantic,
        Self::Reasoning,
        Self::KnowledgeExplanation,
        Self::Mathematics,
        Self::Programming,
        Self::OfficeBusiness,
        Self::Research,
        Self::ToolUse,
    ];

    pub const ALL: [Self; 9] = [
        Self::Semantic,
        Self::Reasoning,
        Self::KnowledgeExplanation,
        Self::Mathematics,
        Self::Programming,
        Self::OfficeBusiness,
        Self::Research,
        Self::ToolUse,
        Self::ParametricCad,
    ];

    #[inline]
    pub const fn bit(self) -> u16 {
        1u16 << (self as u8)
    }

    /// Official ADA-1B General Intelligence scorecard weight (sum across the 8 general domains = 1.00).
    #[inline]
    pub const fn scorecard_weight(self) -> f64 {
        match self {
            Self::Semantic => 0.20,
            Self::Reasoning => 0.20,
            Self::KnowledgeExplanation => 0.15,
            Self::Mathematics => 0.10,
            Self::Programming => 0.10,
            Self::OfficeBusiness => 0.10,
            Self::Research => 0.10,
            Self::ToolUse => 0.05,
            Self::ParametricCad => 0.0, // Evaluated independently as Domain Supremacy
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Reasoning => "reasoning",
            Self::KnowledgeExplanation => "knowledge_explanation",
            Self::Mathematics => "mathematics",
            Self::Programming => "programming",
            Self::OfficeBusiness => "office_business",
            Self::Research => "research",
            Self::ToolUse => "tool_use",
            Self::ParametricCad => "parametric_cad",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "semantic" => Some(Self::Semantic),
            "reasoning" => Some(Self::Reasoning),
            "knowledge_explanation" | "knowledge" => Some(Self::KnowledgeExplanation),
            "mathematics" | "math" => Some(Self::Mathematics),
            "programming" | "coding" => Some(Self::Programming),
            "office_business" | "office" => Some(Self::OfficeBusiness),
            "research" => Some(Self::Research),
            "tool_use" | "tools" => Some(Self::ToolUse),
            "parametric_cad" | "cad" => Some(Self::ParametricCad),
            _ => None,
        }
    }
}

/// The 8 shared, cross-domain transferable SACI computational primitives
/// (one unified primitive bank shared across all 8 general domains + Parametric CAD).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SaciPrimitiveKind {
    SemanticComposition = 0,
    RelationalComputation = 1,
    SequentialComputation = 2,
    QuantitativeTransform = 3,
    CausalRelationships = 4,
    AssociativeRepresentation = 5,
    HierarchicalComposition = 6,
    AbstractTransform = 7,
}

impl SaciPrimitiveKind {
    pub const ALL: [Self; 8] = [
        Self::SemanticComposition,
        Self::RelationalComputation,
        Self::SequentialComputation,
        Self::QuantitativeTransform,
        Self::CausalRelationships,
        Self::AssociativeRepresentation,
        Self::HierarchicalComposition,
        Self::AbstractTransform,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SemanticComposition => "semantic_composition",
            Self::RelationalComputation => "relational_computation",
            Self::SequentialComputation => "sequential_computation",
            Self::QuantitativeTransform => "quantitative_transform",
            Self::CausalRelationships => "causal_relationships",
            Self::AssociativeRepresentation => "associative_representation",
            Self::HierarchicalComposition => "hierarchical_composition",
            Self::AbstractTransform => "abstract_transform",
        }
    }

    /// Bitmask of `AdaCapabilityDomain`s served by this shared SACI primitive.
    #[inline]
    pub const fn transfer_domains_mask(self) -> u16 {
        use AdaCapabilityDomain as D;
        match self {
            Self::SemanticComposition => {
                D::Semantic.bit() | D::Research.bit() | D::OfficeBusiness.bit() | D::ToolUse.bit()
            }
            Self::RelationalComputation => {
                D::ParametricCad.bit()
                    | D::Mathematics.bit()
                    | D::KnowledgeExplanation.bit()
                    | D::Programming.bit()
                    | D::Reasoning.bit()
            }
            Self::SequentialComputation => {
                D::Programming.bit() | D::KnowledgeExplanation.bit() | D::ToolUse.bit()
            }
            Self::QuantitativeTransform => {
                D::Mathematics.bit() | D::OfficeBusiness.bit() | D::ParametricCad.bit()
            }
            Self::CausalRelationships => {
                D::Research.bit() | D::Reasoning.bit() | D::Programming.bit()
            }
            Self::AssociativeRepresentation => D::KnowledgeExplanation.bit() | D::Semantic.bit(),
            Self::HierarchicalComposition => {
                D::OfficeBusiness.bit()
                    | D::Programming.bit()
                    | D::Research.bit()
                    | D::ToolUse.bit()
                    | D::ParametricCad.bit()
            }
            Self::AbstractTransform => D::Reasoning.bit() | D::Mathematics.bit(),
        }
    }

    /// Number of domains that share this primitive (proves cross-domain transfer >= 2).
    #[inline]
    pub const fn transfer_degree(self) -> u32 {
        self.transfer_domains_mask().count_ones()
    }
}

/// Pure-weight `F32` compiled bank of all 8 shared transferable SACI primitives.
#[derive(Clone, Debug)]
pub struct SharedSaciPrimitiveBankF32 {
    pub dim: usize,
    pub quantitative_gate: PureBilinearGateF32,
    pub relational_proj: PureLinearF32,
    pub curve_lut: PureCurveMlpF32,
    pub simulation_canvas: AdaSimulationCanvasWeights,
}

/// Compile the shared transferable SACI primitive bank into pure GPU-safe `F32` weights.
pub fn compile_shared_saci_primitive_bank(
    dim: usize,
) -> Result<SharedSaciPrimitiveBankF32, ErrorCode> {
    if dim < 4 {
        return Err(ErrorCode::InvalidArchitecture);
    }
    let quantitative_gate = compile_saci_blend_gate(AdaBlendMode::AlphaComposite, 3)?;
    let relational_proj = compile_saci_color_matrix(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0],
    )?;
    let curve_lut = compile_saci_tone_curve(&[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])?;
    let simulation_canvas = compile_ada_simulation_canvas(3, dim, 3)?;
    Ok(SharedSaciPrimitiveBankF32 {
        dim,
        quantitative_gate,
        relational_proj,
        curve_lut,
        simulation_canvas,
    })
}

// ---------------------------------------------------------------------------
// 1. ADA 4-Family Neuron / Expert Allocation
// ---------------------------------------------------------------------------

/// Partition of a layer's experts or neurons across ADA's 4 exclusive feature families.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaFamilyBudget {
    pub knowledge_experts: u32,
    pub reasoning_experts: u32,
    pub saci_experts: u32,
    pub simulation_experts: u32,
}

impl AdaFamilyBudget {
    #[inline]
    pub const fn total(&self) -> u32 {
        self.knowledge_experts
            + self.reasoning_experts
            + self.saci_experts
            + self.simulation_experts
    }

    /// Map an expert index `0..total()` to its assigned ADA `FeatureFamily` (O(1), branch-minimal).
    #[inline]
    pub const fn family_of_expert(&self, expert_idx: u32) -> Option<FeatureFamily> {
        let k_end = self.knowledge_experts;
        let r_end = k_end + self.reasoning_experts;
        let s_end = r_end + self.saci_experts;
        let m_end = s_end + self.simulation_experts;
        if expert_idx < k_end {
            Some(FeatureFamily::Knowledge)
        } else if expert_idx < r_end {
            Some(FeatureFamily::Reasoning)
        } else if expert_idx < s_end {
            Some(FeatureFamily::SaciInterface)
        } else if expert_idx < m_end {
            Some(FeatureFamily::Simulation)
        } else {
            None
        }
    }

    /// Default ADA 128-expert allocation: 64 Knowledge, 24 Reasoning, 24 SACI, 16 Simulation.
    pub const fn default_128() -> Self {
        Self {
            knowledge_experts: 64,
            reasoning_experts: 24,
            saci_experts: 24,
            simulation_experts: 16,
        }
    }

    /// Compact local allocation (16 experts): 8 Knowledge, 3 Reasoning, 3 SACI, 2 Simulation.
    pub const fn default_local_16() -> Self {
        Self {
            knowledge_experts: 8,
            reasoning_experts: 3,
            saci_experts: 3,
            simulation_experts: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Pure-Weight SACI Computational Engine Compiler (F32 / BF16, Zero F64)
// ---------------------------------------------------------------------------

/// Blend modes compiled directly into standard SwiGLU / gated bilinear weights.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaBlendMode {
    /// `out = a * b` (exact bilinear product via SwiGLU gate).
    Multiply = 0,
    /// `out = a + b - a * b` (exact Screen blend via linear + SwiGLU gate).
    Screen = 1,
    /// `out = alpha * fg + (1 - alpha) * bg` (exact alpha compositing).
    AlphaComposite = 2,
}

/// Pure-weight linear layer (`y = W x + b`) in GPU-safe `f32`.
#[derive(Clone, Debug)]
pub struct PureLinearF32 {
    pub out_dim: usize,
    pub in_dim: usize,
    /// Row-major `[out_dim, in_dim]`.
    pub weight: Vec<f32>,
    /// `[out_dim]`.
    pub bias: Vec<f32>,
}

impl PureLinearF32 {
    #[inline]
    pub fn forward(&self, x: &[f32], out: &mut [f32]) -> Result<(), ErrorCode> {
        if x.len() != self.in_dim || out.len() != self.out_dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        for r in 0..self.out_dim {
            let row = &self.weight[r * self.in_dim..(r + 1) * self.in_dim];
            let mut acc = self.bias[r];
            for c in 0..self.in_dim {
                acc += row[c] * x[c];
            }
            out[r] = acc;
        }
        Ok(())
    }
}

/// Pure-weight 2D depthwise convolution kernel (`[channels, 1, k, k]`) in GPU-safe `f32`.
#[derive(Clone, Debug)]
pub struct PureConv2dF32 {
    pub channels: usize,
    pub kernel_size: usize,
    /// Contiguous `[channels * kernel_size * kernel_size]`.
    pub weight: Vec<f32>,
}

/// Pure-weight SwiGLU / bilinear gated block:
/// `y = W_down * ((W_gate * x) ⊙ (W_up * x)) + W_skip * x`
/// Executes exact multiplication, Screen blend, and Alpha compositing using only standard
/// matrix multiplications and elementwise Hadamard product (no external engine).
#[derive(Clone, Debug)]
pub struct PureBilinearGateF32 {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub w_skip: Vec<f32>,
}

impl PureBilinearGateF32 {
    pub fn forward(&self, x: &[f32], out: &mut [f32]) -> Result<(), ErrorCode> {
        if x.len() != self.in_dim || out.len() != self.out_dim {
            return Err(ErrorCode::ShapeMismatch);
        }
        let mut h = vec![0.0_f32; self.hidden_dim];
        for r in 0..self.hidden_dim {
            let g_row = &self.w_gate[r * self.in_dim..(r + 1) * self.in_dim];
            let u_row = &self.w_up[r * self.in_dim..(r + 1) * self.in_dim];
            let mut g_acc = 0.0_f32;
            let mut u_acc = 0.0_f32;
            for c in 0..self.in_dim {
                g_acc += g_row[c] * x[c];
                u_acc += u_row[c] * x[c];
            }
            h[r] = g_acc * u_acc;
        }
        for r in 0..self.out_dim {
            let d_row = &self.w_down[r * self.hidden_dim..(r + 1) * self.hidden_dim];
            let s_row = &self.w_skip[r * self.in_dim..(r + 1) * self.in_dim];
            let mut acc = 0.0_f32;
            for c in 0..self.hidden_dim {
                acc += d_row[c] * h[c];
            }
            for c in 0..self.in_dim {
                acc += s_row[c] * x[c];
            }
            out[r] = acc;
        }
        Ok(())
    }
}

/// Pure-weight piecewise-linear ReLU MLP (`y = c0 + sum_k a_k * ReLU(x - knot_k)`),
/// representing exact tone curves, gamma LUTs, levels, and thresholding in `f32`.
#[derive(Clone, Debug)]
pub struct PureCurveMlpF32 {
    pub num_knots: usize,
    pub w_in: Vec<f32>,
    pub b_in: Vec<f32>,
    pub w_out: Vec<f32>,
    pub b_out: f32,
}

impl PureCurveMlpF32 {
    #[inline]
    pub fn forward_scalar(&self, x: f32) -> f32 {
        let mut acc = self.b_out;
        for k in 0..self.num_knots {
            let pre = self.w_in[k] * x + self.b_in[k];
            if pre > 0.0 {
                acc += self.w_out[k] * pre;
            }
        }
        acc
    }
}

/// Compile a 3×3 color-space / exposure / white-balance / contrast transform
/// directly into a pure-weight `PureLinearF32` layer.
pub fn compile_saci_color_matrix(
    matrix_3x3: [[f32; 3]; 3],
    bias_3: [f32; 3],
) -> Result<PureLinearF32, ErrorCode> {
    let mut weight = Vec::with_capacity(9);
    for row in &matrix_3x3 {
        for &v in row {
            if !v.is_finite() {
                return Err(ErrorCode::InvalidInput);
            }
            weight.push(v);
        }
    }
    for &b in &bias_3 {
        if !b.is_finite() {
            return Err(ErrorCode::InvalidInput);
        }
    }
    Ok(PureLinearF32 {
        out_dim: 3,
        in_dim: 3,
        weight,
        bias: bias_3.to_vec(),
    })
}

/// Compile a standard image-processing 3×3 filter (Gaussian blur, sharpen, Sobel edge)
/// into a pure-weight depthwise `PureConv2dF32` kernel across `channels`.
pub fn compile_saci_conv3x3(
    channels: usize,
    kernel_3x3: [f32; 9],
) -> Result<PureConv2dF32, ErrorCode> {
    if channels == 0 || kernel_3x3.iter().any(|v| !v.is_finite()) {
        return Err(ErrorCode::InvalidInput);
    }
    let mut weight = Vec::with_capacity(channels * 9);
    for _ in 0..channels {
        weight.extend_from_slice(&kernel_3x3);
    }
    Ok(PureConv2dF32 {
        channels,
        kernel_size: 3,
        weight,
    })
}

/// Compile an exact Photoshop-style blend mode (`Multiply`, `Screen`, `AlphaComposite`)
/// for `channels` channels into a pure-weight `PureBilinearGateF32` (SwiGLU) block.
///
/// Input layout:
/// - `Multiply` / `Screen`: `[a_0..a_{C-1}, b_0..b_{C-1}]` (`in_dim = 2 * C`, `out_dim = C`)
/// - `AlphaComposite`: `[fg_0..fg_{C-1}, bg_0..bg_{C-1}, alpha]` (`in_dim = 2 * C + 1`, `out_dim = C`)
pub fn compile_saci_blend_gate(
    mode: AdaBlendMode,
    channels: usize,
) -> Result<PureBilinearGateF32, ErrorCode> {
    if channels == 0 {
        return Err(ErrorCode::InvalidInput);
    }
    match mode {
        AdaBlendMode::Multiply => {
            let in_dim = 2 * channels;
            let mut w_gate = vec![0.0_f32; channels * in_dim];
            let mut w_up = vec![0.0_f32; channels * in_dim];
            let mut w_down = vec![0.0_f32; channels * channels];
            let w_skip = vec![0.0_f32; channels * in_dim];
            for c in 0..channels {
                w_gate[c * in_dim + c] = 1.0; // selects a_c
                w_up[c * in_dim + channels + c] = 1.0; // selects b_c
                w_down[c * channels + c] = 1.0; // + a_c * b_c
            }
            Ok(PureBilinearGateF32 {
                in_dim,
                hidden_dim: channels,
                out_dim: channels,
                w_gate,
                w_up,
                w_down,
                w_skip,
            })
        }
        AdaBlendMode::Screen => {
            // out_c = a_c + b_c - a_c * b_c
            let in_dim = 2 * channels;
            let mut w_gate = vec![0.0_f32; channels * in_dim];
            let mut w_up = vec![0.0_f32; channels * in_dim];
            let mut w_down = vec![0.0_f32; channels * channels];
            let mut w_skip = vec![0.0_f32; channels * in_dim];
            for c in 0..channels {
                w_gate[c * in_dim + c] = 1.0;
                w_up[c * in_dim + channels + c] = 1.0;
                w_down[c * channels + c] = -1.0; // - a_c * b_c
                w_skip[c * in_dim + c] = 1.0; // + a_c
                w_skip[c * in_dim + channels + c] = 1.0; // + b_c
            }
            Ok(PureBilinearGateF32 {
                in_dim,
                hidden_dim: channels,
                out_dim: channels,
                w_gate,
                w_up,
                w_down,
                w_skip,
            })
        }
        AdaBlendMode::AlphaComposite => {
            // out_c = bg_c + alpha * (fg_c - bg_c)
            let in_dim = 2 * channels + 1;
            let alpha_idx = 2 * channels;
            let mut w_gate = vec![0.0_f32; channels * in_dim];
            let mut w_up = vec![0.0_f32; channels * in_dim];
            let mut w_down = vec![0.0_f32; channels * channels];
            let mut w_skip = vec![0.0_f32; channels * in_dim];
            for c in 0..channels {
                w_gate[c * in_dim + alpha_idx] = 1.0; // alpha
                w_up[c * in_dim + c] = 1.0; // + fg_c
                w_up[c * in_dim + channels + c] = -1.0; // - bg_c
                w_down[c * channels + c] = 1.0; // + alpha * (fg_c - bg_c)
                w_skip[c * in_dim + channels + c] = 1.0; // + bg_c
            }
            Ok(PureBilinearGateF32 {
                in_dim,
                hidden_dim: channels,
                out_dim: channels,
                w_gate,
                w_up,
                w_down,
                w_skip,
            })
        }
    }
}

/// Compile a strictly increasing piecewise-linear tone curve `[(x_0, y_0), ..., (x_n, y_n)]`
/// directly into an exact 2-layer ReLU MLP (`PureCurveMlpF32`).
pub fn compile_saci_tone_curve(knots: &[(f32, f32)]) -> Result<PureCurveMlpF32, ErrorCode> {
    if knots.len() < 2 {
        return Err(ErrorCode::InvalidInput);
    }
    for w in knots.windows(2) {
        if !w[0].0.is_finite()
            || !w[0].1.is_finite()
            || !w[1].0.is_finite()
            || !w[1].1.is_finite()
            || w[1].0 <= w[0].0
        {
            return Err(ErrorCode::InvalidInput);
        }
    }

    let n = knots.len() - 1;
    let mut w_in = Vec::with_capacity(n);
    let mut b_in = Vec::with_capacity(n);
    let mut w_out = Vec::with_capacity(n);

    let mut prev_slope = 0.0_f32;
    for i in 0..n {
        let (x0, y0) = knots[i];
        let (x1, y1) = knots[i + 1];
        let slope = (y1 - y0) / (x1 - x0);
        let delta_slope = slope - prev_slope;
        w_in.push(1.0);
        b_in.push(-x0);
        w_out.push(delta_slope);
        prev_slope = slope;
    }

    Ok(PureCurveMlpF32 {
        num_knots: n,
        w_in,
        b_in,
        w_out,
        b_out: knots[0].1,
    })
}

// ---------------------------------------------------------------------------
// 3. ADA Internal Simulation Canvas Weights (`Simulation` Feature Family)
// ---------------------------------------------------------------------------

/// Pure-weight internal canvas projection bundle for ADA's `Simulation` (`M`) family.
///
/// - `write_proj`: maps SACI engine output `[engine_dim]` -> canvas state `[canvas_dim]`
/// - `read_proj`: maps canvas state `[canvas_dim]` -> Reasoning observation `[obs_dim]`
/// - `verify_proj`: maps `[obs_dim]` -> scalar verification score for Reasoning halt check
#[derive(Clone, Debug)]
pub struct AdaSimulationCanvasWeights {
    pub engine_dim: usize,
    pub canvas_dim: usize,
    pub obs_dim: usize,
    pub write_proj: PureLinearF32,
    pub read_proj: PureLinearF32,
    pub verify_proj: PureLinearF32,
}

/// Compile an exact isometric/pass-through internal simulation canvas (`canvas_dim >= engine_dim`)
/// so `SACI` writes directly into the internal canvas and `Reasoning` reads it losslessly.
pub fn compile_ada_simulation_canvas(
    engine_dim: usize,
    canvas_dim: usize,
    obs_dim: usize,
) -> Result<AdaSimulationCanvasWeights, ErrorCode> {
    if engine_dim == 0 || canvas_dim < engine_dim || obs_dim == 0 {
        return Err(ErrorCode::InvalidArchitecture);
    }

    // write_proj: embeds engine_dim into first engine_dim slots of canvas_dim
    let mut w_write = vec![0.0_f32; canvas_dim * engine_dim];
    for i in 0..engine_dim {
        w_write[i * engine_dim + i] = 1.0;
    }

    // read_proj: projects canvas_dim -> obs_dim (exact identity on overlapping coordinates)
    let mut w_read = vec![0.0_f32; obs_dim * canvas_dim];
    let shared = obs_dim.min(canvas_dim);
    for i in 0..shared {
        w_read[i * canvas_dim + i] = 1.0;
    }

    // verify_proj: normalized L1/mean energy probe on observation vector
    let scale = 1.0 / (obs_dim as f32);
    let w_verify = vec![scale; obs_dim];

    Ok(AdaSimulationCanvasWeights {
        engine_dim,
        canvas_dim,
        obs_dim,
        write_proj: PureLinearF32 {
            out_dim: canvas_dim,
            in_dim: engine_dim,
            weight: w_write,
            bias: vec![0.0_f32; canvas_dim],
        },
        read_proj: PureLinearF32 {
            out_dim: obs_dim,
            in_dim: canvas_dim,
            weight: w_read,
            bias: vec![0.0_f32; obs_dim],
        },
        verify_proj: PureLinearF32 {
            out_dim: 1,
            in_dim: obs_dim,
            weight: w_verify,
            bias: vec![0.0_f32],
        },
    })
}

// ---------------------------------------------------------------------------
// 4. ADA → General Quintessa Scale Compiler Bridge (Local to 1T+ Burn Weights)
// ---------------------------------------------------------------------------

/// ADA-specific full-model specification combining ADA's 3:1 KDA+MLA hybrid stack
/// with ADA's 4-family exclusive MoE budget (`K`, `R`, `S`, `M`).
#[derive(Clone, Debug)]
pub struct AdaScaleSpec {
    pub name: String,
    pub vocab_size: u64,
    pub hidden_dim: u64,
    pub num_layers: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub mla_kv_latent_dim: u32,
    pub kda_state_dim: u32,
    pub kda_dplr_rank: u32,
    pub family_budget: AdaFamilyBudget,
    pub num_shared_experts: u32,
    pub expert_ffn_dim: u64,
    pub top_k_experts: u32,
    pub etf_num_neurons: u64,
    pub profile: CompileProfile,
    pub max_shard_bytes: u64,
    pub max_memory_mb: u64,
    pub base_seed: u64,
}

impl AdaScaleSpec {
    /// **ADA-1T**: 1.047T-parameter ADA Hybrid MoE (`96` layers, 3:1 KDA+MLA, 128 4-family experts).
    pub fn preset_ada_1t() -> Self {
        Self {
            name: String::from("ada-1t-moe"),
            vocab_size: 131_072,
            hidden_dim: 8_192,
            num_layers: 96,
            attention_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            mla_kv_latent_dim: 1_024,
            kda_state_dim: 8_192,
            kda_dplr_rank: 64,
            family_budget: AdaFamilyBudget::default_128(),
            num_shared_experts: 2,
            expert_ffn_dim: 3_328,
            top_k_experts: 8,
            etf_num_neurons: 16_384,
            profile: CompileProfile::Balanced,
            max_shard_bytes: DEFAULT_MAX_SHARD_BYTES,
            max_memory_mb: 256,
            base_seed: 20_261_000_000_000_000,
        }
    }

    /// **ADA-1B General-Purpose Cognitive LLM**: ~1.05B total parameters (~375M active/token),
    /// 24 layers (3:1 KDA + MLA), 8 shared-primitive MoE experts + 1 shared expert, `hidden_dim = 1024`,
    /// ~`1.95 GiB` in `BF16` — fits directly on an 8 GB RAM / GTX 1650 (4 GB VRAM) machine.
    pub fn preset_ada_1b() -> Self {
        Self {
            name: String::from("ada-1b-general"),
            vocab_size: 49_152,
            hidden_dim: 1_024,
            num_layers: 24,
            attention_heads: 16,
            kv_heads: 4,
            head_dim: 64,
            mla_kv_latent_dim: 256,
            kda_state_dim: 1_024,
            kda_dplr_rank: 16,
            family_budget: AdaFamilyBudget {
                knowledge_experts: 4,
                reasoning_experts: 2,
                saci_experts: 4,
                simulation_experts: 2,
            },
            num_shared_experts: 1,
            expert_ffn_dim: 1_024,
            top_k_experts: 2,
            etf_num_neurons: 2_048,
            profile: CompileProfile::Balanced,
            max_shard_bytes: 1024 * 1024 * 1024, // 1 GiB shards -> 2 clean shards for ~2 GiB 1B model
            max_memory_mb: 128,
            base_seed: 20_260_001_000_000_000,
        }
    }

    /// **ADA-Local-3B**: Laptop/GTX-1650-friendly ~3B ADA Hybrid MoE (~500M active/token, 16 4-family experts).
    pub fn preset_ada_local_3b() -> Self {
        Self {
            name: String::from("ada-local-3b-moe"),
            vocab_size: 32_768,
            hidden_dim: 1_536,
            num_layers: 24,
            attention_heads: 12,
            kv_heads: 4,
            head_dim: 128,
            mla_kv_latent_dim: 256,
            kda_state_dim: 1_536,
            kda_dplr_rank: 16,
            family_budget: AdaFamilyBudget::default_local_16(),
            num_shared_experts: 1,
            expert_ffn_dim: 1_536,
            top_k_experts: 2,
            etf_num_neurons: 2_048,
            profile: CompileProfile::Memory,
            max_shard_bytes: 2 * 1024 * 1024 * 1024,
            max_memory_mb: 128,
            base_seed: 20_260_003_000_000_000,
        }
    }

    /// Lower this ADA-specific spec into Quintessa's general `ScaleModelSpec`
    /// (pure Burn `.safetensors`, GPU-safe no-`F64`, standard learned-style tensor layout).
    pub fn lower_to_scale_spec(&self) -> Result<ScaleModelSpec, ErrorCode> {
        let total_experts = self.family_budget.total();
        if total_experts == 0 {
            return Err(ErrorCode::InvalidArchitecture);
        }
        Ok(ScaleModelSpec {
            name: self.name.clone(),
            vocab_size: self.vocab_size,
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            // Exact ADA 3:1 KDA + MLA hybrid attention topology
            layer_pattern: vec![
                ScaleLayerKind::Kda,
                ScaleLayerKind::Kda,
                ScaleLayerKind::Kda,
                ScaleLayerKind::Mla,
            ],
            attention_heads: self.attention_heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            mla_kv_latent_dim: self.mla_kv_latent_dim,
            kda_state_dim: self.kda_state_dim,
            kda_dplr_rank: self.kda_dplr_rank,
            ffn_dim: self.expert_ffn_dim,
            num_experts: total_experts,
            num_shared_experts: self.num_shared_experts,
            expert_ffn_dim: self.expert_ffn_dim,
            top_k_experts: self.top_k_experts,
            tie_word_embeddings: false,
            include_pondernet: true,
            include_etf_superposition: self.etf_num_neurons > 0,
            etf_num_neurons: self.etf_num_neurons,
            rope_axes: 1,
            include_rectified_flow_head: false,
            audio_codebooks: 0,
            fallback_dtype: WeightDtype::BF16,
            profile: self.profile,
            weight_origin: WeightOrigin::Learned,
            gpu_safe_no_f64: true,
            base_seed: self.base_seed,
            max_shard_bytes: self.max_shard_bytes,
            max_memory_mb: self.max_memory_mb,
        })
    }
}

/// Plan an ADA model (local 3B up to 1T+) through Quintessa's general sharded compiler.
pub fn plan_ada_scale_model(
    spec: &AdaScaleSpec,
    telemetry: &mut TelemetryBus<256>,
) -> Result<ShardedCompilationPlan, ErrorCode> {
    let general_spec = spec.lower_to_scale_spec()?;
    plan_scale_model(&general_spec, telemetry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ada_family_budget_exclusive_partition() {
        let b = AdaFamilyBudget::default_128();
        assert_eq!(b.total(), 128);
        assert_eq!(b.family_of_expert(0), Some(FeatureFamily::Knowledge));
        assert_eq!(b.family_of_expert(63), Some(FeatureFamily::Knowledge));
        assert_eq!(b.family_of_expert(64), Some(FeatureFamily::Reasoning));
        assert_eq!(b.family_of_expert(88), Some(FeatureFamily::SaciInterface));
        assert_eq!(b.family_of_expert(112), Some(FeatureFamily::Simulation));
        assert_eq!(b.family_of_expert(128), None);
    }

    #[test]
    fn pure_weight_saci_photo_engine_ops_exact_in_f32() {
        // 1. Color matrix (warm highlights + slight exposure lift)
        let color = compile_saci_color_matrix(
            [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.9]],
            [0.05, 0.0, -0.02],
        )
        .unwrap();
        let mut rgb_out = [0.0_f32; 3];
        color.forward(&[0.5, 0.4, 0.3], &mut rgb_out).unwrap();
        assert!((rgb_out[0] - 0.60).abs() < 1e-6);
        assert!((rgb_out[1] - 0.40).abs() < 1e-6);
        assert!((rgb_out[2] - 0.25).abs() < 1e-6);

        // 2. SwiGLU Multiply blend: a * b
        let mul = compile_saci_blend_gate(AdaBlendMode::Multiply, 3).unwrap();
        let mut out = [0.0_f32; 3];
        mul.forward(&[0.5, 0.8, 0.25, 0.4, 0.5, 0.8], &mut out)
            .unwrap();
        assert!((out[0] - 0.20).abs() < 1e-6);
        assert!((out[1] - 0.40).abs() < 1e-6);
        assert!((out[2] - 0.20).abs() < 1e-6);

        // 3. SwiGLU Screen blend: a + b - a*b
        let scr = compile_saci_blend_gate(AdaBlendMode::Screen, 3).unwrap();
        scr.forward(&[0.5, 0.25, 0.0, 0.5, 0.4, 0.7], &mut out)
            .unwrap();
        assert!((out[0] - 0.75).abs() < 1e-6);
        assert!((out[1] - 0.55).abs() < 1e-6);
        assert!((out[2] - 0.70).abs() < 1e-6);

        // 4. SwiGLU AlphaComposite: alpha * fg + (1 - alpha) * bg
        let comp = compile_saci_blend_gate(AdaBlendMode::AlphaComposite, 3).unwrap();
        comp.forward(&[1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.25], &mut out)
            .unwrap();
        assert!((out[0] - 0.25).abs() < 1e-6);
        assert!((out[1] - 0.75).abs() < 1e-6);
        assert!((out[2] - 0.50).abs() < 1e-6);

        // 5. Piecewise-linear tone curve ReLU MLP
        let curve = compile_saci_tone_curve(&[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]).unwrap();
        assert!((curve.forward_scalar(0.0) - 0.0).abs() < 1e-6);
        assert!((curve.forward_scalar(0.25) - 0.35).abs() < 1e-6);
        assert!((curve.forward_scalar(0.5) - 0.70).abs() < 1e-6);
        assert!((curve.forward_scalar(0.75) - 0.85).abs() < 1e-6);
        assert!((curve.forward_scalar(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ada_simulation_canvas_write_read_verify_roundtrip() {
        let canvas = compile_ada_simulation_canvas(3, 8, 3).unwrap();
        let saci_pixel = [0.6_f32, 0.4, 0.2];
        let mut canvas_state = [0.0_f32; 8];
        let mut obs = [0.0_f32; 3];
        let mut score = [0.0_f32; 1];

        canvas
            .write_proj
            .forward(&saci_pixel, &mut canvas_state)
            .unwrap();
        canvas.read_proj.forward(&canvas_state, &mut obs).unwrap();
        canvas.verify_proj.forward(&obs, &mut score).unwrap();

        assert_eq!(obs, saci_pixel);
        assert!((score[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn ada_1t_and_local_3b_lower_to_sharded_burn_plans() {
        let mut tel: TelemetryBus<256> = TelemetryBus::new();

        let plan_1t = plan_ada_scale_model(&AdaScaleSpec::preset_ada_1t(), &mut tel).unwrap();
        assert!(plan_1t.total_params >= 1_000_000_000_000);
        assert!(plan_1t.gpu_safe_no_f64);

        let plan_3b = plan_ada_scale_model(&AdaScaleSpec::preset_ada_local_3b(), &mut tel).unwrap();
        assert!(plan_3b.total_params >= 2_500_000_000);
        assert!(plan_3b.active_params_per_token < plan_3b.total_params / 3);
    }
}
