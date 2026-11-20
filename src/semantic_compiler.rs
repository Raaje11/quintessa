//! True Semantic Weight Compilers for Quintessa's 5 Research-Derived Primitives.
//!
//! Translates explicit user-specified behaviors (sparse concept detectors, causal bigram/trigram
//! triggers, exact token-horizon state registers, associative key-value attention routes, and
//! contractive dual-timescale refinement plans) directly into deterministic mathematical weights:
//!
//! 1. `compile_semantic_jumprelu`     (Gemma Scope 2: unit-norm `W_dec`, oblique `W_enc`, calibrated `F32` thresholds)
//! 2. `compile_semantic_short_conv`   (LFM2.5: causal 3-tap `t-2 / t-1 / t` double-gated depthwise conv)
//! 3. `compile_semantic_mamba2`       (Granite 4.0 H: closed-form inverse-softplus `dt_bias` for exact half-lives)
//! 4. `compile_semantic_gated_attn`   (HRM-Text 1B: associative `Q/K/V` routing + high-gain `sigmoid(G)` gate)
//! 5. `compile_semantic_dual_timescale` (HRM-Text 1B: norm-`sqrt(d)` `z_l_init` + contractive `H`/`L` operators)

use crate::primitives::{
    DoubleGatedShortConv, GatedAttentionGqkv, JumpReLuSparseLayer, PonderNetRefinement,
    SelectiveSsmMamba2,
};
use crate::superposition_wsp::{FeatureSpace, RuleLiteral};
use crate::sys::ErrorCode;

// ===========================================================================
// 1. JumpReLU Sparse Concept Compiler (Gemma Scope 2)
// ===========================================================================

/// One semantic concept detector compiled into a JumpReLU sparse neuron:
/// fires when the signed sum of `literals` exceeds `deadzone_threshold`.
#[derive(Clone, Debug)]
pub struct SemanticConceptSpec {
    pub literals: Vec<RuleLiteral>,
    pub output_feature: u32,
    pub deadzone_threshold: f64,
}

/// Compile explicit semantic concepts into a `JumpReLuSparseEncoder`:
/// - `w_dec` rows are exact unit-norm directions from `out_space` (Gemma Scope 2 invariant)
/// - `w_enc` rows use oblique interference-cancelling directions `(U^T U + λI)^{-1} U^T`
///   from `in_space` so active concepts do not bleed into inactive ones
/// - `threshold` stores exact per-concept dead-zone gates (always exported as `F32`)
pub fn compile_semantic_jumprelu(
    in_space: &FeatureSpace,
    out_space: &FeatureSpace,
    concepts: &[SemanticConceptSpec],
    whitening_lambda: f64,
) -> Result<JumpReLuSparseLayer, ErrorCode> {
    if concepts.is_empty() || in_space.dim != out_space.dim {
        return Err(ErrorCode::InvalidInput);
    }
    let d = in_space.dim;
    let m = concepts.len();
    let oblique_in = in_space.compute_oblique_read_directions(whitening_lambda)?;

    // JumpReLuSparseLayer layout: w_enc is [d_in, width]
    let mut w_enc = vec![0.0_f64; d * m];
    let mut b_enc = vec![0.0_f64; m];
    let mut threshold = vec![0.0_f64; m];
    let mut w_dec = vec![0.0_f64; m * d];
    let b_dec = vec![0.0_f64; d];

    for (c_idx, spec) in concepts.iter().enumerate() {
        if spec.literals.is_empty() || !spec.deadzone_threshold.is_finite() {
            return Err(ErrorCode::InvalidInput);
        }
        let mut enc_vec = vec![0.0_f64; d];
        let mut pos_count = 0usize;
        for lit in &spec.literals {
            let u_tilde = oblique_in
                .direction(lit.feature)
                .ok_or(ErrorCode::InvalidInput)?;
            let sign = if lit.sign >= 0 {
                pos_count += 1;
                1.0_f64
            } else {
                -1.0_f64
            };
            for (dst, &u) in enc_vec.iter_mut().zip(u_tilde.iter()) {
                *dst += sign * u;
            }
        }
        // Normalize pre-activation so all positive literals active => dot == 1.0
        let inv_k = 1.0 / (pos_count.max(1) as f64);
        for (i, &v) in enc_vec.iter().enumerate() {
            w_enc[i * m + c_idx] = v * inv_k;
        }
        b_enc[c_idx] = 0.0;
        threshold[c_idx] = spec.deadzone_threshold.max(0.0);

        // Unit-norm decoder direction for output_feature (Gemma Scope 2 invariant)
        let u_out = out_space
            .direction(spec.output_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let dec_row = &mut w_dec[c_idx * d..(c_idx + 1) * d];
        let norm = u_out.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        for (dst, &u) in dec_row.iter_mut().zip(u_out.iter()) {
            *dst = u / norm;
        }
    }

    JumpReLuSparseLayer::new(d, m, d, w_enc, b_enc, threshold, w_dec, b_dec)
}

// ===========================================================================
// 2. Double-Gated Short Conv Bigram/Trigram Compiler (LFM2.5)
// ===========================================================================

/// One causal local-context pattern compiled into LFM2.5's 3-tap depthwise conv:
///   "if `tap_t2` (optional) at t-2, `tap_t1` (optional) at t-1, and `tap_t0` at t -> emit `output_feature`"
#[derive(Clone, Copy, Debug)]
pub struct ShortConvPatternSpec {
    pub tap_t2: Option<u32>,
    pub tap_t1: Option<u32>,
    pub tap_t0: u32,
    pub output_feature: u32,
    pub gain: f64,
}

/// Compile explicit causal n-gram patterns into a `DoubleGatedShortConv` (`L = 3`):
/// Each pattern `p` gets a dedicated channel `ch = p % d`:
/// - `B` gate reads the current token feature `tap_t0` (opens when current token matches)
/// - `X` value carries the causal sequence into the 3-tap depthwise filter:
///   - `tap0` (`t-2`), `tap1` (`t-1`, LFM2.5 bigram mixer), `tap2` (`t`, refinement)
/// - `C` gate reads `tap_t0` (ensuring output fires strictly at the completing token `t`)
/// - `out_proj` writes `gain * u_out[output_feature]` back into `FeatureSpace`.
pub fn compile_semantic_short_conv(
    in_space: &FeatureSpace,
    out_space: &FeatureSpace,
    patterns: &[ShortConvPatternSpec],
    whitening_lambda: f64,
) -> Result<DoubleGatedShortConv, ErrorCode> {
    let d = in_space.dim;
    if patterns.is_empty() || patterns.len() > d || out_space.dim != d {
        return Err(ErrorCode::InvalidInput);
    }
    let k_len = 3usize;
    let oblique_in = in_space.compute_oblique_read_directions(whitening_lambda)?;

    let mut in_proj = vec![0.0_f64; (3 * d) * d];
    let mut conv_weight = vec![0.0_f64; d * k_len];
    let mut out_proj = vec![0.0_f64; d * d];

    for (ch, pat) in patterns.iter().enumerate() {
        if !pat.gain.is_finite() {
            return Err(ErrorCode::InvalidInput);
        }
        let u_t0 = oblique_in
            .direction(pat.tap_t0)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_out = out_space
            .direction(pat.output_feature)
            .ok_or(ErrorCode::InvalidInput)?;

        // Channel layout in `in_proj` [3*d, d]:
        //   row `ch`         -> B gate (current token or previous context gate)
        //   row `d + ch`     -> C gate (opens strictly when tap_t0 is present at step t)
        //   row `2*d + ch`   -> X input stream fed into the causal 3-tap depthwise conv
        let (b_part, rest) = in_proj.split_at_mut(d * d);
        let (c_part, x_part) = rest.split_at_mut(d * d);
        let b_row = &mut b_part[ch * d..(ch + 1) * d];
        let c_row = &mut c_part[ch * d..(ch + 1) * d];
        let x_row = &mut x_part[ch * d..(ch + 1) * d];

        // C gate opens strictly when `tap_t0` is active at the current step `t`.
        c_row.copy_from_slice(u_t0);

        // Build X + B so historical taps pass through the conv buffer cleanly:
        // If `tap_t1` is specified, B & X respond to `tap_t1` (and `tap_t2` if present),
        // while C gates the final convolved output strictly on `tap_t0` at time `t`.
        // DoubleGatedShortConv kernel lag layout: k=0 is current (t), k=1 is lag-1 (t-1), k=2 is lag-2 (t-2).
        let mut active_taps = 0usize;
        if let Some(f2) = pat.tap_t2 {
            let u2 = oblique_in.direction(f2).ok_or(ErrorCode::InvalidInput)?;
            for i in 0..d {
                b_row[i] += u2[i];
                x_row[i] += u2[i];
            }
            conv_weight[ch * k_len + 2] = 1.0; // lag-2 (t-2)
            active_taps += 1;
        }
        if let Some(f1) = pat.tap_t1 {
            let u1 = oblique_in.direction(f1).ok_or(ErrorCode::InvalidInput)?;
            for i in 0..d {
                b_row[i] += u1[i];
                x_row[i] += u1[i];
            }
            conv_weight[ch * k_len + 1] = 1.0; // lag-1 (t-1, LFM2.5 bigram mixer)
            active_taps += 1;
        }
        if active_taps == 0 {
            // Pure self-refinement pattern on lag-0 (t, LFM2.5 deep layer semantics)
            b_row.copy_from_slice(u_t0);
            x_row.copy_from_slice(u_t0);
            conv_weight[ch * k_len] = 1.0; // lag-0 (t)
        } else {
            let inv = 1.0 / (active_taps as f64);
            conv_weight[ch * k_len + 1] *= inv;
            conv_weight[ch * k_len + 2] *= inv;
        }

        // Write channel `ch` into `output_feature` direction in `out_proj` [d, d]
        for r in 0..d {
            out_proj[r * d + ch] = pat.gain * u_out[r];
        }
    }

    DoubleGatedShortConv::new(d, k_len, in_proj, conv_weight, out_proj)
}

// ===========================================================================
// 3. Selective SSM Mamba-2 State-Horizon Compiler (Granite 4.0 H)
// ===========================================================================

/// One semantic memory register compiled into a Mamba-2 head with exact retention half-life.
#[derive(Clone, Copy, Debug)]
pub struct Mamba2RegisterSpec {
    pub head_idx: usize,
    /// Target memory half-life in tokens (`decay = 0.5^(1 / half_life_tokens)`).
    pub half_life_tokens: f64,
    /// Direct residual skip weight `D[head_idx]`.
    pub skip_gain: f64,
}

/// Exact inverse softplus: given `y > 0`, returns `x` such that `ln(1 + exp(x)) == y`.
#[inline]
pub fn inverse_softplus(y: f64) -> f64 {
    if y > 20.0 {
        y
    } else if y <= 1e-12 {
        -27.631
    } else {
        y.exp_m1().ln()
    }
}

/// Compile exact token-horizon retention schedules into a `SelectiveSsmMamba2` block.
/// Heads specified in `registers` get closed-form `dt_bias` so that at `dt_raw = 0`,
/// `decay_h = exp(-softplus(dt_bias_h) * exp(a_log_h)) == 0.5^(1 / half_life_tokens)`.
/// Unspecified heads receive Granite 4.0 H's log-spaced multi-timescale bank `[-9.0, +3.0]`.
pub fn compile_semantic_mamba2(
    n_heads: usize,
    d_head: usize,
    d_state: usize,
    registers: &[Mamba2RegisterSpec],
) -> Result<SelectiveSsmMamba2, ErrorCode> {
    if n_heads == 0 || d_head == 0 || d_state == 0 {
        return Err(ErrorCode::InvalidInput);
    }

    let a_log = vec![0.0_f64; n_heads]; // exp(0.0) == 1.0 for clean closed-form dt calibration
    let mut d_skip = vec![0.03_f64; n_heads];
    let mut dt_bias: Vec<f64> = (0..n_heads)
        .map(|h| {
            let frac = if n_heads > 1 {
                (h as f64) / ((n_heads - 1) as f64)
            } else {
                0.5
            };
            -9.0 + 12.0 * frac
        })
        .collect();

    for reg in registers {
        if reg.head_idx >= n_heads
            || !reg.half_life_tokens.is_finite()
            || reg.half_life_tokens <= 0.0
            || !reg.skip_gain.is_finite()
        {
            return Err(ErrorCode::InvalidInput);
        }
        let dt_target = std::f64::consts::LN_2 / reg.half_life_tokens;
        dt_bias[reg.head_idx] = inverse_softplus(dt_target).clamp(-12.0, 6.0);
        d_skip[reg.head_idx] = reg.skip_gain;
    }

    Ok(SelectiveSsmMamba2 {
        n_heads,
        d_head,
        d_state,
        a_log,
        d_skip,
        dt_bias,
    })
}

// ===========================================================================
// 4. Gated Attention GQKV Routing Compiler (HRM-Text 1B)
// ===========================================================================

/// One associative attention route compiled into `GatedAttentionGqkv`:
/// When token `t` has `query_feature`, head `head_idx` attends to past tokens with `key_feature`,
/// reads `value_feature`, gates by `sigmoid(gate_strength * query_act)`, and writes `output_feature`.
#[derive(Clone, Copy, Debug)]
pub struct AttnRouteSpec {
    pub head_idx: usize,
    pub query_feature: u32,
    pub key_feature: u32,
    pub value_feature: u32,
    pub output_feature: u32,
    /// HRM-Text 1B empirical finding: `G` gate has ~3x higher gain than `Q/K/V`.
    pub gate_strength: f64,
}

/// Compile explicit associative key-value lookup & selective gating circuits into `GatedAttentionGqkv`.
pub fn compile_semantic_gated_attn(
    in_space: &FeatureSpace,
    out_space: &FeatureSpace,
    num_heads: usize,
    routes: &[AttnRouteSpec],
    whitening_lambda: f64,
) -> Result<GatedAttentionGqkv, ErrorCode> {
    let d = in_space.dim;
    if num_heads == 0 || !d.is_multiple_of(num_heads) || out_space.dim != d || routes.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }
    let head_dim = d / num_heads;
    let oblique_in = in_space.compute_oblique_read_directions(whitening_lambda)?;

    let mut gqkv_proj = vec![0.0_f64; (4 * d) * d];
    let mut o_proj = vec![0.0_f64; d * d];
    let mut head_slot_used = vec![0usize; num_heads];
    let qk_scale = (head_dim as f64).sqrt().sqrt() * 2.5; // crisp softmax routing temperature

    for route in routes {
        if route.head_idx >= num_heads || !route.gate_strength.is_finite() {
            return Err(ErrorCode::InvalidInput);
        }
        let slot = head_slot_used[route.head_idx];
        if slot >= head_dim {
            return Err(ErrorCode::BufferTooSmall);
        }
        head_slot_used[route.head_idx] += 1;

        let channel = route.head_idx * head_dim + slot;
        let u_q = oblique_in
            .direction(route.query_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_k = oblique_in
            .direction(route.key_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_v = oblique_in
            .direction(route.value_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_out = out_space
            .direction(route.output_feature)
            .ok_or(ErrorCode::InvalidInput)?;

        // G row [channel]: high-gain selective gate driven by query_feature (HRM 3x singular value)
        let g_row = &mut gqkv_proj[channel * d..(channel + 1) * d];
        for (dst, &u) in g_row.iter_mut().zip(u_q.iter()) {
            *dst = route.gate_strength * u;
        }
        // Q row [d + channel]
        let q_row = &mut gqkv_proj[(d + channel) * d..(d + channel + 1) * d];
        for (dst, &u) in q_row.iter_mut().zip(u_q.iter()) {
            *dst = qk_scale * u;
        }
        // K row [2*d + channel]
        let k_row = &mut gqkv_proj[(2 * d + channel) * d..(2 * d + channel + 1) * d];
        for (dst, &u) in k_row.iter_mut().zip(u_k.iter()) {
            *dst = qk_scale * u;
        }
        // V row [3*d + channel]
        let v_row = &mut gqkv_proj[(3 * d + channel) * d..(3 * d + channel + 1) * d];
        v_row.copy_from_slice(u_v);

        // O projection [d, d]: column `channel` writes into `output_feature`
        // Compensate for sigmoid(gate_strength) so active route outputs unit amplitude.
        let sig_active = 1.0 / (1.0 + (-route.gate_strength).exp());
        let out_gain = if sig_active > 1e-6 {
            1.0 / sig_active
        } else {
            1.0
        };
        for r in 0..d {
            o_proj[r * d + channel] = out_gain * u_out[r];
        }
    }

    Ok(GatedAttentionGqkv {
        dim: d,
        num_heads,
        head_dim,
        gqkv_proj,
        o_proj,
    })
}

// ===========================================================================
// 5. Dual-Timescale Contractive Refinement Compiler (HRM-Text 1B)
// ===========================================================================

/// One slow-plan (`z_H`) to fast-execution (`z_L`) contractive refinement coupling.
#[derive(Clone, Copy, Debug)]
pub struct DualTimescaleCouplingSpec {
    pub plan_feature: u32,
    pub fast_feature: u32,
    /// Strictly contractive spectral gain in `(0.0, 0.95)` guaranteeing convergence.
    pub contraction_rate: f64,
}

/// Compiled dual-timescale operators (`z_l_init`, `w_l_step`, `w_h_step`).
#[derive(Clone, Debug)]
pub struct CompiledDualTimescale {
    pub dim: usize,
    pub z_l_init: Vec<f64>,
    pub w_l_step: Vec<f64>,
    pub w_h_step: Vec<f64>,
}

/// Compile provably contractive dual-timescale (`H` slow plan, `L` fast refinement) operators:
/// - `||z_l_init||_2 == sqrt(d)` (exact HRM-Text 1B empirical invariant)
/// - `w_l_step` maps `plan_feature -> fast_feature` with contraction `< 0.95`
/// - `w_h_step` integrates `fast_feature -> plan_feature` with contraction `< 0.95`
pub fn compile_semantic_dual_timescale(
    space: &FeatureSpace,
    couplings: &[DualTimescaleCouplingSpec],
    whitening_lambda: f64,
) -> Result<CompiledDualTimescale, ErrorCode> {
    let d = space.dim;
    if d == 0 || couplings.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }
    let oblique = space.compute_oblique_read_directions(whitening_lambda)?;

    let mut z_l_init = vec![1.0_f64; d]; // exact L2 norm == sqrt(d)
    let mut w_l_step = vec![0.0_f64; d * d];
    let mut w_h_step = vec![0.0_f64; d * d];

    // If couplings are present, align z_l_init with the sum of fast_features, rescaled to sqrt(d)
    let mut init_accum = vec![0.0_f64; d];
    for c in couplings {
        if !c.contraction_rate.is_finite()
            || c.contraction_rate <= 0.0
            || c.contraction_rate >= 0.95
        {
            return Err(ErrorCode::InvalidInput);
        }
        let u_plan = space
            .direction(c.plan_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_plan_read = oblique
            .direction(c.plan_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_fast = space
            .direction(c.fast_feature)
            .ok_or(ErrorCode::InvalidInput)?;
        let u_fast_read = oblique
            .direction(c.fast_feature)
            .ok_or(ErrorCode::InvalidInput)?;

        for i in 0..d {
            init_accum[i] += u_fast[i];
            let l_row = &mut w_l_step[i * d..(i + 1) * d];
            let h_row = &mut w_h_step[i * d..(i + 1) * d];
            let uf_i = c.contraction_rate * u_fast[i];
            let up_i = c.contraction_rate * u_plan[i];
            for j in 0..d {
                l_row[j] += uf_i * u_plan_read[j];
                h_row[j] += up_i * u_fast_read[j];
            }
        }
    }
    let norm = init_accum.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-12 {
        let target_norm = (d as f64).sqrt();
        for i in 0..d {
            z_l_init[i] = init_accum[i] * (target_norm / norm);
        }
    }

    Ok(CompiledDualTimescale {
        dim: d,
        z_l_init,
        w_l_step,
        w_h_step,
    })
}

// ===========================================================================
// 6. PonderNet Geometric Refinement & Verification Compiler (ADA-o1)
// ===========================================================================

/// Semantic specification for a PonderNet geometric refinement + verification block.
#[derive(Clone, Copy, Debug)]
pub struct PonderNetSpec {
    pub t_cap: usize,
    pub lambda_prior: f64,
    /// Feature index whose presence triggers halting (`sigmoid(w_halt · ws + b_halt) >= 0.5`).
    pub halt_feature: u32,
    /// Feature index whose presence signals verified confidence (`sigmoid(w_verify · ws + b_verify) >= 0.5`).
    pub verify_feature: u32,
    /// Contractive feedback gain in `[0.0, 0.9)` applied to `ws_{t-1}` into `ctx_t`.
    pub feedback_gain: f64,
}

/// Compile explicit halting, verification, and contractive feedback semantics into `PonderNetRefinement`.
pub fn compile_semantic_pondernet(
    space: &FeatureSpace,
    spec: &PonderNetSpec,
    whitening_lambda: f64,
) -> Result<PonderNetRefinement, ErrorCode> {
    let d = space.dim;
    if d == 0
        || spec.t_cap == 0
        || !spec.lambda_prior.is_finite()
        || !spec.feedback_gain.is_finite()
        || spec.feedback_gain < 0.0
        || spec.feedback_gain >= 0.95
    {
        return Err(ErrorCode::InvalidInput);
    }
    let oblique = space.compute_oblique_read_directions(whitening_lambda)?;
    let u_halt = oblique
        .direction(spec.halt_feature)
        .ok_or(ErrorCode::InvalidInput)?;
    let u_verify = oblique
        .direction(spec.verify_feature)
        .ok_or(ErrorCode::InvalidInput)?;

    // Diagonal contractive feedback operator `feedback_gain * I_d`
    let mut w_feedback = vec![0.0_f64; d * d];
    for i in 0..d {
        w_feedback[i * d + i] = spec.feedback_gain;
    }

    // High-gain oblique halt/verify gates centered at threshold 0.5
    let gain = 6.0_f64;
    let w_halt: Vec<f64> = u_halt.iter().map(|&v| gain * v).collect();
    let w_verify: Vec<f64> = u_verify.iter().map(|&v| gain * v).collect();

    Ok(PonderNetRefinement {
        dim: d,
        t_cap: spec.t_cap,
        lambda_prior: spec.lambda_prior,
        w_feedback,
        w_halt,
        b_halt: -0.5 * gain,
        w_verify,
        b_verify: -0.5 * gain,
    })
}
