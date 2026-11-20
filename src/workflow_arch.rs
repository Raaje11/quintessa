//! Configurable Custom Model Architecture & Intelligent Mixed-Precision Workflow Compiler
//! (`workflow_arch.rs`).
//!
//! Compiles custom architectures created by a Workflow UI or `.qwf` spec file across all
//! researched primitives (NumericThresholds, LinearScorer, SuperpositionRules with optional
//! oblique whitened encoder, DecisionTree, JumpReLuSparse, DoubleGatedShortConv,
//! SelectiveSsmMamba2, GatedAttentionGqkv, DualTimescaleRecurrence, ResidualTrainable).
//!
//! Supports `CompileProfile` (`Memory`, `Latency`, `Accuracy`, `Balanced`) to automatically
//! select per-feature storage datatypes (`F64`/`F32`/`BF16`/`F16`/`I8`), runs the 5-axis
//! Quintessa Verifier + Pre-Emission Weight-Space Health Gate, and emits strictly `.safetensors`.

use std::path::Path;

use crate::compiler_passes::{promote_outlier_dtype, DEFAULT_OUTLIER_PROMOTE_ABS};
use crate::safetensors_writer::{
    write_mixed_safetensors, TypedNamedTensor, WeightDtype, WeightHealthSummary,
};
use crate::semantic_compiler::{
    compile_semantic_dual_timescale, compile_semantic_gated_attn, compile_semantic_jumprelu,
    compile_semantic_mamba2, compile_semantic_pondernet, compile_semantic_short_conv,
    AttnRouteSpec, DualTimescaleCouplingSpec, Mamba2RegisterSpec, PonderNetSpec,
    SemanticConceptSpec, ShortConvPatternSpec,
};
use crate::superposition_wsp::{
    DecisionTreeNode, FeatureSpace, LinearScorerLayer, LinearScorerSpec, NumericThresholdLayer,
    NumericThresholdSpec, RuleLayer, RuleLiteral, RuleSpec, SplitMix64, ThresholdOp,
};
use crate::sys::ErrorCode;
use crate::verifier::{verify_superposition_layer, VerifierReport, VerifyConfig};

/// Intelligent compilation target profile controlling per-feature dtypes, oblique whitening,
/// and verifier strictness.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileProfile {
    /// Minimize `.safetensors` footprint (`BF16`/`F16` bulk matrices, `F32` thresholds/norms, `F64` exact math).
    Memory = 0,
    /// Minimize inference latency (`BF16` bulk + sharper JumpReLU sparsity + early cycle halt).
    Latency = 1,
    /// Maximize numerical fidelity (`F32` bulk + oblique whitened rule encoders + `F64` exact math).
    Accuracy = 2,
    /// Balanced mixed-precision (`BF16` bulk projections, `F32` thresholds/biases/norms/states, `F64` exact math).
    Balanced = 3,
}

impl CompileProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Latency => "latency",
            Self::Accuracy => "accuracy",
            Self::Balanced => "balanced",
        }
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "memory" | "mem" => Some(Self::Memory),
            "latency" | "fast" => Some(Self::Latency),
            "accuracy" | "acc" => Some(Self::Accuracy),
            "balanced" | "default" => Some(Self::Balanced),
            _ => None,
        }
    }

    /// Select optimal storage `WeightDtype` for a semantic tensor role.
    #[inline]
    pub fn select_dtype(self, role: TensorSemanticRole, fallback: WeightDtype) -> WeightDtype {
        match role {
            TensorSemanticRole::ExactMath => WeightDtype::F64,
            TensorSemanticRole::ThresholdOrBiasOrState => WeightDtype::F32,
            TensorSemanticRole::BulkMatrix | TensorSemanticRole::ConvKernel => match self {
                Self::Memory => WeightDtype::F16,
                Self::Latency | Self::Balanced => WeightDtype::BF16,
                Self::Accuracy => fallback,
            },
            TensorSemanticRole::FeatureDictionary => match self {
                Self::Memory => WeightDtype::F16,
                Self::Latency | Self::Balanced => WeightDtype::BF16,
                Self::Accuracy => WeightDtype::F32,
            },
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorSemanticRole {
    ExactMath = 0,
    ThresholdOrBiasOrState = 1,
    BulkMatrix = 2,
    ConvKernel = 3,
    FeatureDictionary = 4,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreezePolicy {
    /// Compiled software logic kept frozen during any downstream fine-tuning.
    Frozen = 0,
    /// Compiled logic lightly trainable with L2 anchor pull toward compiled weights.
    AnchoredL2 = 1,
    /// Residual trainable capacity initialized near zero (§8).
    Trainable = 2,
}

impl FreezePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            FreezePolicy::Frozen => "frozen",
            FreezePolicy::AnchoredL2 => "anchored_l2",
            FreezePolicy::Trainable => "trainable",
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    NumericThresholds = 0,
    LinearScorer = 1,
    SuperpositionRules = 2,
    DecisionTree = 3,
    ResidualTrainable = 4,
    JumpReLuSparse = 5,
    DoubleGatedShortConv = 6,
    SelectiveSsmMamba2 = 7,
    GatedAttentionGqkv = 8,
    DualTimescaleRecurrence = 9,
    PonderNetRefinement = 10,
}

#[derive(Clone, Debug)]
pub enum BlockPayload {
    NumericThresholds {
        num_inputs: usize,
        specs: Vec<NumericThresholdSpec>,
    },
    LinearScorer {
        specs: Vec<LinearScorerSpec>,
    },
    SuperpositionRules {
        rules: Vec<RuleSpec>,
        tau: f64,
    },
    DecisionTree {
        tree: DecisionTreeNode,
        tau: f64,
    },
    ResidualTrainable {
        hidden_units: usize,
        init_scale: f64,
        seed: u64,
    },
    /// Gemma Scope 2 style unit-norm `w_dec` + oblique `w_enc` + `F32` JumpReLU thresholds.
    JumpReLuSparse {
        width: usize,
        threshold_val: f64,
        seed: u64,
        concepts: Vec<SemanticConceptSpec>,
    },
    /// LFM2.5 style double-gated short causal conv (`in_proj [3d, d]`, `conv [d, 1, L]`, `out_proj [d, d]`).
    DoubleGatedShortConv {
        l_cache: usize,
        depth_ratio: f64,
        seed: u64,
        patterns: Vec<ShortConvPatternSpec>,
    },
    /// Granite 4.0 H style Mamba-2 selective SSM with logarithmic multi-timescale `dt_bias`.
    SelectiveSsmMamba2 {
        n_heads: usize,
        d_head: usize,
        d_state: usize,
        seed: u64,
        registers: Vec<Mamba2RegisterSpec>,
    },
    /// HRM-Text 1B style gated attention (`gqkv_proj [4d, d]` + `o_proj [d, d]`) + parameterless RMSNorm.
    GatedAttentionGqkv {
        num_heads: usize,
        seed: u64,
        routes: Vec<AttnRouteSpec>,
    },
    /// HRM-Text 1B style nested `H_cycles x L_cycles` recurrence (`z_L_init` scaled to `sqrt(d)`).
    DualTimescaleRecurrence {
        h_cycles: usize,
        l_cycles: usize,
        seed: u64,
        couplings: Vec<DualTimescaleCouplingSpec>,
    },
    /// ADA-o1 style PonderNet geometric halting + verification-gated workspace refinement.
    PonderNetRefinement {
        spec: PonderNetSpec,
    },
}

#[derive(Clone, Debug)]
pub struct WorkflowBlock {
    pub id: u32,
    pub kind: BlockKind,
    pub freeze: FreezePolicy,
    pub payload: BlockPayload,
}

/// Complete configurable custom architecture created by a Workflow UI or `.qwf` file.
#[derive(Clone, Debug)]
pub struct WorkflowSpec {
    pub name: String,
    pub hidden_dim: usize,
    pub num_features: usize,
    pub num_outputs: usize,
    pub input_seed: u64,
    pub output_seed: u64,
    pub one_sentinel_feature: Option<u32>,
    pub dtype: WeightDtype,
    pub profile: CompileProfile,
    pub blocks: Vec<WorkflowBlock>,
}

/// Compiled artifacts, 5-axis verification reports, and pre-emission weight health summary.
#[derive(Clone, Debug)]
pub struct CompiledWorkflowResult {
    pub bytes_written: u64,
    pub total_tensors: usize,
    pub total_weights: usize,
    pub rule_reports: Vec<VerifierReport>,
    pub weight_health: WeightHealthSummary,
}

struct OwnedTypedTensor {
    name: String,
    shape: Vec<usize>,
    dtype: WeightDtype,
    data: Vec<f64>,
}

/// Compile a custom `WorkflowSpec`, run the 5-axis verifier on every compiled rule/tree layer,
/// enforce the pre-emission weight-space health gate, and emit strictly to `.safetensors`.
pub fn compile_workflow_to_safetensors(
    spec: &WorkflowSpec,
    verify_cfg: &VerifyConfig,
    out_path: &Path,
) -> Result<CompiledWorkflowResult, ErrorCode> {
    if out_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(ErrorCode::InvalidInput);
    }
    if spec.hidden_dim == 0 || spec.num_features == 0 || spec.num_outputs == 0 {
        return Err(ErrorCode::InvalidArchitecture);
    }

    let in_space = FeatureSpace::new_gaussian(spec.num_features, spec.hidden_dim, spec.input_seed)?;
    let out_space =
        FeatureSpace::new_gaussian(spec.num_outputs, spec.hidden_dim, spec.output_seed)?;

    let dict_dtype = spec
        .profile
        .select_dtype(TensorSemanticRole::FeatureDictionary, spec.dtype);
    let bulk_dtype = spec
        .profile
        .select_dtype(TensorSemanticRole::BulkMatrix, spec.dtype);
    let conv_dtype = spec
        .profile
        .select_dtype(TensorSemanticRole::ConvKernel, spec.dtype);
    let state_dtype = spec
        .profile
        .select_dtype(TensorSemanticRole::ThresholdOrBiasOrState, spec.dtype);

    let mut owned: Vec<OwnedTypedTensor> = Vec::with_capacity(spec.blocks.len() * 4 + 2);
    owned.push(OwnedTypedTensor {
        name: "space.input.u".to_string(),
        shape: vec![in_space.num_features, in_space.dim],
        dtype: dict_dtype,
        data: in_space.directions.clone(),
    });
    owned.push(OwnedTypedTensor {
        name: "space.output.u".to_string(),
        shape: vec![out_space.num_features, out_space.dim],
        dtype: dict_dtype,
        data: out_space.directions.clone(),
    });

    let mut rule_reports = Vec::new();
    let use_oblique = spec.profile == CompileProfile::Accuracy && spec.hidden_dim <= 512;

    for block in &spec.blocks {
        match &block.payload {
            BlockPayload::NumericThresholds { num_inputs, specs } => {
                let layer = NumericThresholdLayer::compile(*num_inputs, &in_space, specs)?;
                let mut params = Vec::with_capacity(specs.len() * 4);
                for sp in specs {
                    params.push(sp.input_idx as f64);
                    params.push(sp.threshold);
                    params.push(sp.steepness);
                    params.push(match sp.op {
                        ThresholdOp::Greater => 1.0,
                        ThresholdOp::Less => -1.0,
                    });
                }
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.threshold.params", block.id),
                    shape: vec![specs.len(), 4],
                    dtype: state_dtype,
                    data: params,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.threshold.write_dirs", block.id),
                    shape: vec![specs.len(), layer.dim],
                    dtype: dict_dtype,
                    data: layer.write_dirs,
                });
            }
            BlockPayload::LinearScorer { specs } => {
                let layer = LinearScorerLayer::compile(&in_space, specs)?;
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.scorer.w", block.id),
                    shape: vec![layer.num_scores, layer.dim],
                    dtype: bulk_dtype,
                    data: layer.w,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.scorer.b", block.id),
                    shape: vec![layer.num_scores],
                    dtype: state_dtype,
                    data: layer.bias,
                });
            }
            BlockPayload::SuperpositionRules { rules, tau } => {
                let layer = if use_oblique {
                    RuleLayer::compile_oblique(
                        &in_space,
                        &out_space,
                        rules,
                        *tau,
                        spec.one_sentinel_feature,
                        0.5,
                    )?
                } else {
                    RuleLayer::compile(
                        &in_space,
                        &out_space,
                        rules,
                        *tau,
                        spec.one_sentinel_feature,
                    )?
                };
                let report = verify_superposition_layer(
                    &in_space,
                    &out_space,
                    &layer,
                    rules,
                    spec.one_sentinel_feature,
                    None,
                    verify_cfg,
                )?;
                rule_reports.push(report);
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.rules.w", block.id),
                    shape: vec![layer.num_rules, layer.in_dim],
                    dtype: bulk_dtype,
                    data: layer.w,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.rules.b", block.id),
                    shape: vec![layer.num_rules],
                    dtype: state_dtype,
                    data: layer.b,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.rules.v_rows", block.id),
                    shape: vec![layer.num_rules, layer.out_dim],
                    dtype: dict_dtype,
                    data: layer.v_rows,
                });
            }
            BlockPayload::DecisionTree { tree, tau } => {
                let mut rules = Vec::new();
                tree.compile_to_rules(&mut rules);
                let layer = RuleLayer::compile(
                    &in_space,
                    &out_space,
                    &rules,
                    *tau,
                    spec.one_sentinel_feature,
                )?;
                let report = verify_superposition_layer(
                    &in_space,
                    &out_space,
                    &layer,
                    &rules,
                    spec.one_sentinel_feature,
                    None,
                    verify_cfg,
                )?;
                rule_reports.push(report);
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.tree.w", block.id),
                    shape: vec![layer.num_rules, layer.in_dim],
                    dtype: bulk_dtype,
                    data: layer.w,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.tree.b", block.id),
                    shape: vec![layer.num_rules],
                    dtype: state_dtype,
                    data: layer.b,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.tree.v_rows", block.id),
                    shape: vec![layer.num_rules, layer.out_dim],
                    dtype: dict_dtype,
                    data: layer.v_rows,
                });
            }
            BlockPayload::ResidualTrainable {
                hidden_units,
                init_scale,
                seed,
            } => {
                if *hidden_units == 0 || !init_scale.is_finite() || *init_scale < 0.0 {
                    return Err(ErrorCode::InvalidInput);
                }
                let mut rng = SplitMix64::new(*seed);
                let mut w_in = vec![0.0_f64; hidden_units * spec.hidden_dim];
                for v in &mut w_in {
                    *v = rng.next_normal() * *init_scale;
                }
                let b_in = vec![0.0_f64; *hidden_units];
                let w_out = vec![0.0_f64; hidden_units * spec.hidden_dim];
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.residual.w_in", block.id),
                    shape: vec![*hidden_units, spec.hidden_dim],
                    dtype: bulk_dtype,
                    data: w_in,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.residual.b_in", block.id),
                    shape: vec![*hidden_units],
                    dtype: state_dtype,
                    data: b_in,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.residual.w_out", block.id),
                    shape: vec![*hidden_units, spec.hidden_dim],
                    dtype: bulk_dtype,
                    data: w_out,
                });
            }
            BlockPayload::JumpReLuSparse {
                width,
                threshold_val,
                seed,
                concepts,
            } => {
                let d = spec.hidden_dim;
                let (w_enc, b_enc, thr, w_dec, b_dec, eff_width) = if !concepts.is_empty() {
                    let compiled =
                        compile_semantic_jumprelu(&in_space, &out_space, concepts, 1e-3)?;
                    (
                        compiled.w_enc,
                        compiled.b_enc,
                        compiled.threshold,
                        compiled.w_dec,
                        compiled.b_dec,
                        compiled.width,
                    )
                } else {
                    let dec_space = FeatureSpace::new_gaussian(*width, d, *seed)?;
                    let mut w_enc = vec![0.0_f64; d * *width];
                    for j in 0..*width {
                        let row = &dec_space.directions[j * d..(j + 1) * d];
                        for i in 0..d {
                            w_enc[i * *width + j] = row[i];
                        }
                    }
                    (
                        w_enc,
                        vec![-0.25 * *threshold_val; *width],
                        vec![*threshold_val; *width],
                        dec_space.directions,
                        vec![0.0_f64; d],
                        *width,
                    )
                };
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.jumprelu.w_enc", block.id),
                    shape: vec![d, eff_width],
                    dtype: bulk_dtype,
                    data: w_enc,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.jumprelu.b_enc", block.id),
                    shape: vec![eff_width],
                    dtype: state_dtype,
                    data: b_enc,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.jumprelu.threshold", block.id),
                    shape: vec![eff_width],
                    dtype: state_dtype,
                    data: thr,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.jumprelu.w_dec", block.id),
                    shape: vec![eff_width, d],
                    dtype: dict_dtype,
                    data: w_dec,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.jumprelu.b_dec", block.id),
                    shape: vec![d],
                    dtype: state_dtype,
                    data: b_dec,
                });
            }
            BlockPayload::DoubleGatedShortConv {
                l_cache,
                depth_ratio,
                seed,
                patterns,
            } => {
                let d = spec.hidden_dim;
                let (in_proj, kernel, out_proj, l) = if !patterns.is_empty() {
                    let compiled =
                        compile_semantic_short_conv(&in_space, &out_space, patterns, 1e-3)?;
                    (
                        compiled.in_proj,
                        compiled.kernel,
                        compiled.out_proj,
                        compiled.l_cache,
                    )
                } else {
                    let l = (*l_cache).max(1);
                    let mut rng = SplitMix64::new(*seed);
                    let proj_scale = 0.02;
                    let mut in_proj = vec![0.0_f64; 3 * d * d];
                    for v in &mut in_proj {
                        *v = rng.next_normal() * proj_scale;
                    }
                    let mut out_proj = vec![0.0_f64; d * d];
                    for v in &mut out_proj {
                        *v = rng.next_normal() * proj_scale;
                    }
                    let dr = depth_ratio.clamp(0.0, 1.0);
                    let mut kernel = vec![0.0_f64; d * l];
                    for c in 0..d {
                        for k in 0..l {
                            let gain = if l == 3 {
                                match k {
                                    0 => 0.03 + 0.17 * dr,
                                    1 => 0.21 * (1.0 - 0.8 * dr),
                                    _ => 0.04 * (1.0 - 0.8 * dr),
                                }
                            } else {
                                0.12 / (l as f64)
                            };
                            kernel[c * l + k] = rng.next_normal() * gain;
                        }
                    }
                    (in_proj, kernel, out_proj, l)
                };
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.short_conv.in_proj", block.id),
                    shape: vec![3 * d, d],
                    dtype: bulk_dtype,
                    data: in_proj,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.short_conv.kernel", block.id),
                    shape: vec![d, 1, l],
                    dtype: conv_dtype,
                    data: kernel,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.short_conv.out_proj", block.id),
                    shape: vec![d, d],
                    dtype: bulk_dtype,
                    data: out_proj,
                });
            }
            BlockPayload::SelectiveSsmMamba2 {
                n_heads,
                d_head,
                d_state,
                seed,
                registers,
            } => {
                if *n_heads == 0 || *d_head == 0 || *d_state == 0 {
                    return Err(ErrorCode::InvalidInput);
                }
                let (a_log, d_skip, dt_bias) = if !registers.is_empty() {
                    let compiled = compile_semantic_mamba2(*n_heads, *d_head, *d_state, registers)?;
                    (compiled.a_log, compiled.d_skip, compiled.dt_bias)
                } else {
                    let mut rng = SplitMix64::new(*seed);
                    let a_log: Vec<f64> = (0..*n_heads).map(|_| rng.next_normal() * 0.3).collect();
                    let d_skip = vec![0.03_f64; *n_heads];
                    let dt_bias: Vec<f64> = (0..*n_heads)
                        .map(|h| {
                            let frac = if *n_heads > 1 {
                                (h as f64) / ((*n_heads - 1) as f64)
                            } else {
                                0.5
                            };
                            -9.0 + 12.0 * frac
                        })
                        .collect();
                    (a_log, d_skip, dt_bias)
                };
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.mamba2.a_log", block.id),
                    shape: vec![*n_heads],
                    dtype: state_dtype,
                    data: a_log,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.mamba2.d_skip", block.id),
                    shape: vec![*n_heads],
                    dtype: state_dtype,
                    data: d_skip,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.mamba2.dt_bias", block.id),
                    shape: vec![*n_heads],
                    dtype: state_dtype,
                    data: dt_bias,
                });
            }
            BlockPayload::GatedAttentionGqkv {
                num_heads,
                seed,
                routes,
            } => {
                let d = spec.hidden_dim;
                if *num_heads == 0 || !d.is_multiple_of(*num_heads) {
                    return Err(ErrorCode::InvalidInput);
                }
                let (gqkv, o_proj) = if !routes.is_empty() {
                    let compiled = compile_semantic_gated_attn(
                        &in_space, &out_space, *num_heads, routes, 1e-3,
                    )?;
                    (compiled.gqkv_proj, compiled.o_proj)
                } else {
                    let mut rng = SplitMix64::new(*seed);
                    let scale = 1.0 / (d as f64).sqrt();
                    let gqkv: Vec<f64> =
                        (0..4 * d * d).map(|_| rng.next_normal() * scale).collect();
                    let o_proj: Vec<f64> = (0..d * d).map(|_| rng.next_normal() * scale).collect();
                    (gqkv, o_proj)
                };
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.gated_attn.gqkv_proj", block.id),
                    shape: vec![4 * d, d],
                    dtype: bulk_dtype,
                    data: gqkv,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.gated_attn.o_proj", block.id),
                    shape: vec![d, d],
                    dtype: bulk_dtype,
                    data: o_proj,
                });
            }
            BlockPayload::DualTimescaleRecurrence {
                h_cycles,
                l_cycles,
                seed,
                couplings,
            } => {
                if *h_cycles == 0 || *l_cycles == 0 {
                    return Err(ErrorCode::InvalidInput);
                }
                let d = spec.hidden_dim;
                let ctrl = vec![*h_cycles as f64, *l_cycles as f64];
                if !couplings.is_empty() {
                    let compiled = compile_semantic_dual_timescale(&in_space, couplings, 1e-3)?;
                    owned.push(OwnedTypedTensor {
                        name: format!("block.{}.dual_timescale.z_l_init", block.id),
                        shape: vec![d],
                        dtype: state_dtype,
                        data: compiled.z_l_init,
                    });
                    owned.push(OwnedTypedTensor {
                        name: format!("block.{}.dual_timescale.w_l_step", block.id),
                        shape: vec![d, d],
                        dtype: bulk_dtype,
                        data: compiled.w_l_step,
                    });
                    owned.push(OwnedTypedTensor {
                        name: format!("block.{}.dual_timescale.w_h_step", block.id),
                        shape: vec![d, d],
                        dtype: bulk_dtype,
                        data: compiled.w_h_step,
                    });
                } else {
                    let mut rng = SplitMix64::new(*seed);
                    let z_l_init: Vec<f64> =
                        (0..d).map(|_| rng.next_normal().clamp(-3.0, 3.0)).collect();
                    owned.push(OwnedTypedTensor {
                        name: format!("block.{}.dual_timescale.z_l_init", block.id),
                        shape: vec![d],
                        dtype: state_dtype,
                        data: z_l_init,
                    });
                }
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.dual_timescale.cycles", block.id),
                    shape: vec![2],
                    dtype: state_dtype,
                    data: ctrl,
                });
            }
            BlockPayload::PonderNetRefinement { spec: pspec } => {
                let d = spec.hidden_dim;
                let compiled = compile_semantic_pondernet(&in_space, pspec, 1e-3)?;
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.pondernet.w_feedback", block.id),
                    shape: vec![d, d],
                    dtype: bulk_dtype,
                    data: compiled.w_feedback,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.pondernet.w_halt", block.id),
                    shape: vec![d],
                    dtype: state_dtype,
                    data: compiled.w_halt,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.pondernet.w_verify", block.id),
                    shape: vec![d],
                    dtype: state_dtype,
                    data: compiled.w_verify,
                });
                owned.push(OwnedTypedTensor {
                    name: format!("block.{}.pondernet.gates_meta", block.id),
                    shape: vec![4],
                    dtype: state_dtype,
                    data: vec![
                        compiled.t_cap as f64,
                        compiled.lambda_prior,
                        compiled.b_halt,
                        compiled.b_verify,
                    ],
                });
            }
        }
    }

    // Cap 5: Dynamic Outlier Auto-Promotion & GPU-safe F64 -> F32 cap before emission.
    for t in &mut owned {
        let guard = promote_outlier_dtype(t.dtype, &t.data, DEFAULT_OUTLIER_PROMOTE_ABS, true)?;
        t.dtype = guard.final_dtype;
    }

    // Pure learned Burn (Rust) header with zero compiler fingerprints.
    let metadata = [("format", "burn")];

    let named: Vec<TypedNamedTensor<'_>> = owned
        .iter()
        .map(|t| TypedNamedTensor {
            name: t.name.as_str(),
            shape: t.shape.clone(),
            dtype: t.dtype,
            data: t.data.as_slice(),
        })
        .collect();
    let total_weights: usize = named.iter().map(|t| t.data.len()).sum();
    let total_tensors = named.len();

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let (bytes_written, weight_health) = write_mixed_safetensors(out_path, &metadata, &named)
        .map_err(|_| ErrorCode::InvalidInput)?;

    Ok(CompiledWorkflowResult {
        bytes_written,
        total_tensors,
        total_weights,
        rule_reports,
        weight_health,
    })
}

/// Parse a declarative `.qwf` workflow architecture file (cold-path) into a `WorkflowSpec`.
pub fn parse_workflow_spec(src: &str) -> Result<WorkflowSpec, ErrorCode> {
    let mut name = String::from("quintessa_custom");
    let mut hidden_dim = 512usize;
    let mut num_features = 64usize;
    let mut num_outputs = 4usize;
    let mut input_seed = 2u64;
    let mut output_seed = 3u64;
    let mut one_sentinel_feature = None;
    let mut dtype = WeightDtype::F32;
    let mut profile = CompileProfile::Balanced;

    let mut section = String::new();
    let mut thresh_inputs = 0usize;
    let mut thresholds: Vec<NumericThresholdSpec> = Vec::new();
    let mut scorers: Vec<LinearScorerSpec> = Vec::new();
    let mut rules_tau = 0.5_f64;
    let mut rules: Vec<RuleSpec> = Vec::new();
    let mut residual_units: Option<(usize, f64, u64)> = None;
    let mut jumprelu_cfg: Option<(usize, f64, u64)> = None;
    let mut jumprelu_concepts: Vec<SemanticConceptSpec> = Vec::new();
    let mut short_conv_cfg: Option<(usize, f64, u64)> = None;
    let mut short_conv_patterns: Vec<ShortConvPatternSpec> = Vec::new();
    let mut mamba2_cfg: Option<(usize, usize, usize, u64)> = None;
    let mut mamba2_registers: Vec<Mamba2RegisterSpec> = Vec::new();
    let mut gated_attn_cfg: Option<(usize, u64)> = None;
    let mut gated_attn_routes: Vec<AttnRouteSpec> = Vec::new();
    let mut dual_ts_cfg: Option<(usize, usize, u64)> = None;
    let mut dual_ts_couplings: Vec<DualTimescaleCouplingSpec> = Vec::new();
    let mut pondernet_cfg: Option<PonderNetSpec> = None;

    for raw in src.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_ascii_lowercase();
            continue;
        }

        match section.as_str() {
            "workflow" => {
                let (k, v) = line.split_once('=').ok_or(ErrorCode::InvalidInput)?;
                let (k, v) = (k.trim(), v.trim());
                match k {
                    "name" => name = v.to_string(),
                    "hidden_dim" => hidden_dim = v.parse().map_err(|_| ErrorCode::InvalidInput)?,
                    "num_features" => {
                        num_features = v.parse().map_err(|_| ErrorCode::InvalidInput)?
                    }
                    "num_outputs" => {
                        num_outputs = v.parse().map_err(|_| ErrorCode::InvalidInput)?
                    }
                    "input_seed" => input_seed = v.parse().map_err(|_| ErrorCode::InvalidInput)?,
                    "output_seed" => {
                        output_seed = v.parse().map_err(|_| ErrorCode::InvalidInput)?
                    }
                    "one_sentinel_feature" => {
                        one_sentinel_feature =
                            Some(v.parse::<u32>().map_err(|_| ErrorCode::InvalidInput)?);
                    }
                    "dtype" => {
                        dtype = match v.to_ascii_lowercase().as_str() {
                            "f32" => WeightDtype::F32,
                            "f64" => WeightDtype::F64,
                            "bf16" => WeightDtype::BF16,
                            "f16" => WeightDtype::F16,
                            "i8" => WeightDtype::I8,
                            _ => return Err(ErrorCode::InvalidInput),
                        };
                    }
                    "profile" => {
                        profile = CompileProfile::from_str_opt(v).ok_or(ErrorCode::InvalidInput)?;
                    }
                    _ => {}
                }
            }
            "thresholds" => {
                if let Some((k, v)) = line.split_once('=') {
                    if k.trim() == "num_inputs" {
                        thresh_inputs = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?;
                    }
                } else {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() != 5 {
                        return Err(ErrorCode::InvalidInput);
                    }
                    let op = match parts[3].to_ascii_lowercase().as_str() {
                        "gt" | ">" | ">=" => ThresholdOp::Greater,
                        "lt" | "<" | "<=" => ThresholdOp::Less,
                        _ => return Err(ErrorCode::InvalidInput),
                    };
                    thresholds.push(NumericThresholdSpec {
                        input_idx: parts[0].parse().map_err(|_| ErrorCode::InvalidInput)?,
                        threshold: parts[1].parse().map_err(|_| ErrorCode::InvalidInput)?,
                        steepness: parts[2].parse().map_err(|_| ErrorCode::InvalidInput)?,
                        op,
                        feature_idx: parts[4].parse().map_err(|_| ErrorCode::InvalidInput)?,
                    });
                }
            }
            "scorer" => {
                let (b_str, terms_str) = line.split_once(';').ok_or(ErrorCode::InvalidInput)?;
                let bias: f64 = b_str.trim().parse().map_err(|_| ErrorCode::InvalidInput)?;
                let mut terms = Vec::new();
                for pair in terms_str.split(',') {
                    let pair = pair.trim();
                    if pair.is_empty() {
                        continue;
                    }
                    let (f_s, c_s) = pair.split_once(':').ok_or(ErrorCode::InvalidInput)?;
                    terms.push((
                        f_s.trim()
                            .parse::<u32>()
                            .map_err(|_| ErrorCode::InvalidInput)?,
                        c_s.trim()
                            .parse::<f64>()
                            .map_err(|_| ErrorCode::InvalidInput)?,
                    ));
                }
                scorers.push(LinearScorerSpec { terms, bias });
            }
            "rules" => {
                if let Some((k, v)) = line.split_once('=') {
                    if k.trim() == "tau" {
                        rules_tau = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?;
                    }
                } else {
                    let (lhs, rhs) = line.split_once("->").ok_or(ErrorCode::InvalidInput)?;
                    let output: u32 = rhs.trim().parse().map_err(|_| ErrorCode::InvalidInput)?;
                    let mut literals = Vec::new();
                    for tok in lhs.split(',') {
                        let tok = tok.trim();
                        if tok.is_empty() {
                            continue;
                        }
                        let (sign, num_str) = if let Some(rest) = tok.strip_prefix('-') {
                            (-1i8, rest)
                        } else if let Some(rest) = tok.strip_prefix('+') {
                            (1i8, rest)
                        } else {
                            (1i8, tok)
                        };
                        literals.push(RuleLiteral {
                            feature: num_str
                                .trim()
                                .parse::<u32>()
                                .map_err(|_| ErrorCode::InvalidInput)?,
                            sign,
                        });
                    }
                    rules.push(RuleSpec { literals, output });
                }
            }
            "residual" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = residual_units.get_or_insert((64, 1e-4, 99));
                    match k.trim() {
                        "hidden_units" => {
                            cur.0 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "init_scale" => {
                            cur.1 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "seed" => cur.2 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        _ => {}
                    }
                }
            }
            "jumprelu" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = jumprelu_cfg.get_or_insert((256, 0.5, 101));
                    match k.trim() {
                        "width" => cur.0 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        "threshold" => {
                            cur.1 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "seed" => cur.2 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        "concept" => {
                            // Format: `+1,-2 -> 0 @ 0.45` (or `+1,-2 -> 0`, defaulting threshold to cur.1)
                            let (lhs, rhs) = v.split_once("->").ok_or(ErrorCode::InvalidInput)?;
                            let (out_s, thr_val) = if let Some((o, t)) = rhs.split_once('@') {
                                (
                                    o.trim(),
                                    t.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                                )
                            } else {
                                (rhs.trim(), cur.1)
                            };
                            let output_feature: u32 =
                                out_s.parse().map_err(|_| ErrorCode::InvalidInput)?;
                            let mut literals = Vec::new();
                            for tok in lhs.split(',') {
                                let t = tok.trim();
                                if t.is_empty() {
                                    continue;
                                }
                                let (sign, rest) = if let Some(r) = t.strip_prefix('+') {
                                    (1i8, r)
                                } else if let Some(r) = t.strip_prefix('-') {
                                    (-1i8, r)
                                } else {
                                    (1i8, t)
                                };
                                let f: u32 =
                                    rest.trim().parse().map_err(|_| ErrorCode::InvalidInput)?;
                                literals.push(RuleLiteral { feature: f, sign });
                            }
                            jumprelu_concepts.push(SemanticConceptSpec {
                                literals,
                                output_feature,
                                deadzone_threshold: thr_val,
                            });
                        }
                        _ => {}
                    }
                }
            }
            "short_conv" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = short_conv_cfg.get_or_insert((3, 0.0, 202));
                    match k.trim() {
                        "l_cache" | "kernel_size" => {
                            cur.0 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "depth_ratio" => {
                            cur.1 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "seed" => cur.2 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        "pattern" => {
                            // Format: `f_prev, f_curr -> out * gain` (2 taps) or `f_t2, f_t1, f_t0 -> out * gain`
                            let (lhs, rhs) = v.split_once("->").ok_or(ErrorCode::InvalidInput)?;
                            let (out_s, gain) = if let Some((o, g)) = rhs.split_once('*') {
                                (
                                    o.trim(),
                                    g.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                                )
                            } else {
                                (rhs.trim(), 1.0_f64)
                            };
                            let output_feature: u32 =
                                out_s.parse().map_err(|_| ErrorCode::InvalidInput)?;
                            let taps: Result<Vec<u32>, _> =
                                lhs.split(',').map(|s| s.trim().parse::<u32>()).collect();
                            let taps = taps.map_err(|_| ErrorCode::InvalidInput)?;
                            let (tap_t2, tap_t1, tap_t0) = match taps.as_slice() {
                                [a] => (None, None, *a),
                                [a, b] => (None, Some(*a), *b),
                                [a, b, c] => (Some(*a), Some(*b), *c),
                                _ => return Err(ErrorCode::InvalidInput),
                            };
                            short_conv_patterns.push(ShortConvPatternSpec {
                                tap_t2,
                                tap_t1,
                                tap_t0,
                                output_feature,
                                gain,
                            });
                        }
                        _ => {}
                    }
                }
            }
            "mamba2" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = mamba2_cfg.get_or_insert((16, 32, 64, 303));
                    match k.trim() {
                        "n_heads" | "num_heads" => {
                            cur.0 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "d_head" => {
                            cur.1 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "d_state" | "state_dim" => {
                            cur.2 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "seed" => cur.3 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        "register" => {
                            // Format: `head_idx, half_life_tokens, skip_gain`
                            let parts: Vec<&str> = v.split(',').map(|s| s.trim()).collect();
                            if parts.len() < 2 || parts.len() > 3 {
                                return Err(ErrorCode::InvalidInput);
                            }
                            let head_idx: usize =
                                parts[0].parse().map_err(|_| ErrorCode::InvalidInput)?;
                            let half_life_tokens: f64 =
                                parts[1].parse().map_err(|_| ErrorCode::InvalidInput)?;
                            let skip_gain: f64 = if parts.len() == 3 {
                                parts[2].parse().map_err(|_| ErrorCode::InvalidInput)?
                            } else {
                                0.03
                            };
                            mamba2_registers.push(Mamba2RegisterSpec {
                                head_idx,
                                half_life_tokens,
                                skip_gain,
                            });
                        }
                        _ => {}
                    }
                }
            }
            "gated_attn" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = gated_attn_cfg.get_or_insert((8, 404));
                    match k.trim() {
                        "num_heads" => {
                            cur.0 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "seed" => cur.1 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        "route" => {
                            // Format: `head : q, k, v -> out * gate_strength`
                            let (head_s, rest) =
                                v.split_once(':').ok_or(ErrorCode::InvalidInput)?;
                            let head_idx: usize =
                                head_s.trim().parse().map_err(|_| ErrorCode::InvalidInput)?;
                            let (qkv_s, rhs) =
                                rest.split_once("->").ok_or(ErrorCode::InvalidInput)?;
                            let (out_s, gate_strength) = if let Some((o, g)) = rhs.split_once('*') {
                                (
                                    o.trim(),
                                    g.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                                )
                            } else {
                                (rhs.trim(), 4.0_f64)
                            };
                            let qkv: Result<Vec<u32>, _> =
                                qkv_s.split(',').map(|s| s.trim().parse::<u32>()).collect();
                            let qkv = qkv.map_err(|_| ErrorCode::InvalidInput)?;
                            if qkv.len() != 3 {
                                return Err(ErrorCode::InvalidInput);
                            }
                            let output_feature: u32 =
                                out_s.parse().map_err(|_| ErrorCode::InvalidInput)?;
                            gated_attn_routes.push(AttnRouteSpec {
                                head_idx,
                                query_feature: qkv[0],
                                key_feature: qkv[1],
                                value_feature: qkv[2],
                                output_feature,
                                gate_strength,
                            });
                        }
                        _ => {}
                    }
                }
            }
            "dual_timescale" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = dual_ts_cfg.get_or_insert((2, 3, 505));
                    match k.trim() {
                        "h_cycles" => {
                            cur.0 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "l_cycles" => {
                            cur.1 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "seed" => cur.2 = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                        "coupling" => {
                            // Format: `plan_feature -> fast_feature * contraction_rate`
                            let (lhs, rhs) = v.split_once("->").ok_or(ErrorCode::InvalidInput)?;
                            let (fast_s, rate) = if let Some((f, r)) = rhs.split_once('*') {
                                (
                                    f.trim(),
                                    r.trim().parse().map_err(|_| ErrorCode::InvalidInput)?,
                                )
                            } else {
                                (rhs.trim(), 0.4_f64)
                            };
                            dual_ts_couplings.push(DualTimescaleCouplingSpec {
                                plan_feature: lhs
                                    .trim()
                                    .parse()
                                    .map_err(|_| ErrorCode::InvalidInput)?,
                                fast_feature: fast_s
                                    .parse()
                                    .map_err(|_| ErrorCode::InvalidInput)?,
                                contraction_rate: rate,
                            });
                        }
                        _ => {}
                    }
                }
            }
            "pondernet" => {
                if let Some((k, v)) = line.split_once('=') {
                    let cur = pondernet_cfg.get_or_insert(PonderNetSpec {
                        t_cap: 4,
                        lambda_prior: 0.4,
                        halt_feature: 0,
                        verify_feature: 1,
                        feedback_gain: 0.25,
                    });
                    match k.trim() {
                        "t_cap" => {
                            cur.t_cap = v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "lambda_prior" => {
                            cur.lambda_prior =
                                v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "halt_feature" => {
                            cur.halt_feature =
                                v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "verify_feature" => {
                            cur.verify_feature =
                                v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        "feedback_gain" => {
                            cur.feedback_gain =
                                v.trim().parse().map_err(|_| ErrorCode::InvalidInput)?
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    let mut blocks = Vec::new();
    let mut next_id = 0u32;
    if !thresholds.is_empty() {
        let n_in = thresh_inputs.max(
            thresholds
                .iter()
                .map(|t| t.input_idx as usize + 1)
                .max()
                .unwrap_or(1),
        );
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::NumericThresholds,
            freeze: FreezePolicy::Frozen,
            payload: BlockPayload::NumericThresholds {
                num_inputs: n_in,
                specs: thresholds,
            },
        });
        next_id += 1;
    }
    if !scorers.is_empty() {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::LinearScorer,
            freeze: FreezePolicy::AnchoredL2,
            payload: BlockPayload::LinearScorer { specs: scorers },
        });
        next_id += 1;
    }
    if !rules.is_empty() {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::SuperpositionRules,
            freeze: FreezePolicy::Frozen,
            payload: BlockPayload::SuperpositionRules {
                rules,
                tau: rules_tau,
            },
        });
        next_id += 1;
    }
    if let Some((width, threshold_val, seed)) = jumprelu_cfg {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::JumpReLuSparse,
            freeze: FreezePolicy::AnchoredL2,
            payload: BlockPayload::JumpReLuSparse {
                width,
                threshold_val,
                seed,
                concepts: jumprelu_concepts,
            },
        });
        next_id += 1;
    }
    if let Some((l_cache, depth_ratio, seed)) = short_conv_cfg {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::DoubleGatedShortConv,
            freeze: FreezePolicy::AnchoredL2,
            payload: BlockPayload::DoubleGatedShortConv {
                l_cache,
                depth_ratio,
                seed,
                patterns: short_conv_patterns,
            },
        });
        next_id += 1;
    }
    if let Some((n_heads, d_head, d_state, seed)) = mamba2_cfg {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::SelectiveSsmMamba2,
            freeze: FreezePolicy::AnchoredL2,
            payload: BlockPayload::SelectiveSsmMamba2 {
                n_heads,
                d_head,
                d_state,
                seed,
                registers: mamba2_registers,
            },
        });
        next_id += 1;
    }
    if let Some((num_heads, seed)) = gated_attn_cfg {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::GatedAttentionGqkv,
            freeze: FreezePolicy::AnchoredL2,
            payload: BlockPayload::GatedAttentionGqkv {
                num_heads,
                seed,
                routes: gated_attn_routes,
            },
        });
        next_id += 1;
    }
    if let Some((h_cycles, l_cycles, seed)) = dual_ts_cfg {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::DualTimescaleRecurrence,
            freeze: FreezePolicy::Frozen,
            payload: BlockPayload::DualTimescaleRecurrence {
                h_cycles,
                l_cycles,
                seed,
                couplings: dual_ts_couplings,
            },
        });
        next_id += 1;
    }
    if let Some(pspec) = pondernet_cfg {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::PonderNetRefinement,
            freeze: FreezePolicy::AnchoredL2,
            payload: BlockPayload::PonderNetRefinement { spec: pspec },
        });
        next_id += 1;
    }
    if let Some((hidden_units, init_scale, seed)) = residual_units {
        blocks.push(WorkflowBlock {
            id: next_id,
            kind: BlockKind::ResidualTrainable,
            freeze: FreezePolicy::Trainable,
            payload: BlockPayload::ResidualTrainable {
                hidden_units,
                init_scale,
                seed,
            },
        });
    }

    Ok(WorkflowSpec {
        name,
        hidden_dim,
        num_features,
        num_outputs,
        input_seed,
        output_seed,
        one_sentinel_feature,
        dtype,
        profile,
        blocks,
    })
}
