//! Architecture-Independent Quintessa Computational Primitives (`primitives.rs`).
//!
//! Extracted from Milestone 1–2 archaeology across four distinct reference architectures:
//! 1. `JumpReLuSparseLayer` — Gemma Scope 2 (`w_enc`, `b_enc`, `threshold`, unit-norm `w_dec`, `b_dec`)
//! 2. `DoubleGatedShortConv` — LFM2.5 (`in_proj [3d, d]` -> `[B_gate, C_gate, x]`, `conv [d, L]`, `out_proj [d, d]`)
//! 3. `SelectiveSsmMamba2` — Granite 4.0 H (per-head `A_log`, `D`, `dt_bias`, `H_t = decay * H_{t-1} + dt * (B ⊗ x)`)
//! 4. `DualTimescaleCycleController` — HRM-Text 1B (nested `H_cycles x L_cycles` recurrent state injection with `z_L_init`)
//!
//! All hot-path step functions are strictly zero-allocation and operate on preallocated state/scratch buffers.

use crate::sys::ErrorCode;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveKind {
    JumpReLuSparse = 0,
    DoubleGatedShortConv = 1,
    SelectiveSsmMamba2 = 2,
    DualTimescaleRecurrence = 3,
    GatedAttentionGqkv = 4,
    PonderNetRefinement = 5,
}

// ===========================================================================
// 1. JumpReLU Sparse Superposition Autoencoder / Transcoder (Gemma Scope 2)
// ===========================================================================

/// Sparse feature encoder/decoder with per-feature JumpReLU dead-zone thresholds:
///   `pre = x @ W_enc + b_enc`
///   `z_j = pre_j if pre_j > threshold_j else 0`
///   `y   = z @ W_dec + b_dec`  (sparse SAXPY over active `z_j > 0`)
#[derive(Clone, Debug)]
pub struct JumpReLuSparseLayer {
    pub d_in: usize,
    pub width: usize,
    pub d_out: usize,
    /// Row-major `[d_in, width]`
    pub w_enc: Vec<f64>,
    /// Length `width` (`F32`/`F64` precision required due to large negative biases)
    pub b_enc: Vec<f64>,
    /// Length `width` positive dead-zone thresholds
    pub threshold: Vec<f64>,
    /// Row-major `[width, d_out]` (unit-norm rows for SAE)
    pub w_dec: Vec<f64>,
    /// Length `d_out`
    pub b_dec: Vec<f64>,
}

impl JumpReLuSparseLayer {
    pub fn new(
        d_in: usize,
        width: usize,
        d_out: usize,
        w_enc: Vec<f64>,
        b_enc: Vec<f64>,
        threshold: Vec<f64>,
        w_dec: Vec<f64>,
        b_dec: Vec<f64>,
    ) -> Result<Self, ErrorCode> {
        if d_in == 0
            || width == 0
            || d_out == 0
            || w_enc.len() != d_in * width
            || b_enc.len() != width
            || threshold.len() != width
            || w_dec.len() != width * d_out
            || b_dec.len() != d_out
        {
            return Err(ErrorCode::ShapeMismatch);
        }
        Ok(Self {
            d_in,
            width,
            d_out,
            w_enc,
            b_enc,
            threshold,
            w_dec,
            b_dec,
        })
    }

    /// Zero-allocation forward pass. Writes sparse activations into `z_acts[..width]` and
    /// reconstructed vector into `y_out[..d_out]`. Returns `L0` (number of active features).
    #[inline]
    pub fn forward_into(
        &self,
        x: &[f64],
        z_acts: &mut [f64],
        y_out: &mut [f64],
    ) -> Result<usize, ErrorCode> {
        if x.len() < self.d_in || z_acts.len() < self.width || y_out.len() < self.d_out {
            return Err(ErrorCode::BufferTooSmall);
        }
        let z = &mut z_acts[..self.width];
        z.copy_from_slice(&self.b_enc);
        for (i, &xi) in x[..self.d_in].iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = &self.w_enc[i * self.width..(i + 1) * self.width];
            for (dst, &w) in z.iter_mut().zip(row.iter()) {
                *dst += xi * w;
            }
        }

        let y = &mut y_out[..self.d_out];
        y.copy_from_slice(&self.b_dec);

        let mut l0 = 0usize;
        for (j, zj) in z.iter_mut().enumerate() {
            if *zj > self.threshold[j] {
                let act = *zj;
                l0 += 1;
                let dec_row = &self.w_dec[j * self.d_out..(j + 1) * self.d_out];
                for (dst, &dw) in y.iter_mut().zip(dec_row.iter()) {
                    *dst += act * dw;
                }
            } else {
                *zj = 0.0;
            }
        }
        Ok(l0)
    }
}

// ===========================================================================
// 2. Double-Gated Short Causal Convolution (LFM2.5)
// ===========================================================================

/// Bounded-cache state for `DoubleGatedShortConv` (`[dim, l_cache]`, O(1) memory per token).
#[derive(Clone, Debug)]
pub struct ShortConvCache {
    pub dim: usize,
    pub l_cache: usize,
    /// Circular ring buffer `[l_cache, dim]` holding recent `B_gate * x` vectors.
    pub ring: Vec<f64>,
    pub cursor: usize,
    pub proj_buf: Vec<f64>,
    pub conv_out: Vec<f64>,
}

impl ShortConvCache {
    pub fn new(dim: usize, l_cache: usize) -> Self {
        Self {
            dim,
            l_cache: l_cache.max(1),
            ring: vec![0.0_f64; l_cache.max(1) * dim],
            cursor: 0,
            proj_buf: vec![0.0_f64; 3 * dim],
            conv_out: vec![0.0_f64; dim],
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.cursor = 0;
    }
}

/// LFM2.5 double-gated short causal convolution:
///   `[B, C, x_in] = x @ in_proj^T`
///   `u_t = B * x_in`
///   `v_t = sum_{k=0}^{L-1} kernel[d, k] * u_{t-k}`
///   `y_t = (C * v_t) @ out_proj^T`
#[derive(Clone, Debug)]
pub struct DoubleGatedShortConv {
    pub dim: usize,
    pub l_cache: usize,
    /// Row-major `[3 * dim, dim]` (`B_gate`, `C_gate`, `x_in`)
    pub in_proj: Vec<f64>,
    /// Row-major `[dim, l_cache]` depthwise causal kernel (lag `0` = current, `l_cache-1` = oldest)
    pub kernel: Vec<f64>,
    /// Row-major `[dim, dim]`
    pub out_proj: Vec<f64>,
}

impl DoubleGatedShortConv {
    pub fn new(
        dim: usize,
        l_cache: usize,
        in_proj: Vec<f64>,
        kernel: Vec<f64>,
        out_proj: Vec<f64>,
    ) -> Result<Self, ErrorCode> {
        if dim == 0
            || l_cache == 0
            || in_proj.len() != 3 * dim * dim
            || kernel.len() != dim * l_cache
            || out_proj.len() != dim * dim
        {
            return Err(ErrorCode::ShapeMismatch);
        }
        Ok(Self {
            dim,
            l_cache,
            in_proj,
            kernel,
            out_proj,
        })
    }

    /// Single-timestep causal streaming step (zero heap allocation, O(1) state memory).
    pub fn step_into(
        &self,
        x: &[f64],
        cache: &mut ShortConvCache,
        y_out: &mut [f64],
    ) -> Result<(), ErrorCode> {
        let d = self.dim;
        let l = self.l_cache;
        if x.len() < d || y_out.len() < d || cache.dim < d || cache.l_cache < l {
            return Err(ErrorCode::BufferTooSmall);
        }

        // 1. in_proj: [3d, d] @ x[d] -> proj_buf[3d] = [B, C, x_in]
        for r in 0..(3 * d) {
            let row = &self.in_proj[r * d..(r + 1) * d];
            let mut acc = 0.0_f64;
            for (&w, &xv) in row.iter().zip(x[..d].iter()) {
                acc += w * xv;
            }
            cache.proj_buf[r] = acc;
        }

        // 2. Write B * x_in into ring buffer at current cursor
        let slot_off = cache.cursor * d;
        for i in 0..d {
            let b_g = cache.proj_buf[i];
            let x_in = cache.proj_buf[2 * d + i];
            cache.ring[slot_off + i] = b_g * x_in;
        }

        // 3. Causal depthwise conv over last `l` ring entries, then post-gate by C
        for i in 0..d {
            let mut acc = 0.0_f64;
            for k in 0..l {
                let ring_idx = (cache.cursor + l - k) % l;
                acc += self.kernel[i * l + k] * cache.ring[ring_idx * d + i];
            }
            let c_g = cache.proj_buf[d + i];
            cache.conv_out[i] = c_g * acc;
        }
        cache.cursor = (cache.cursor + 1) % l;

        // 4. out_proj: [d, d] @ conv_out[d] -> y_out[d]
        for r in 0..d {
            let row = &self.out_proj[r * d..(r + 1) * d];
            let mut acc = 0.0_f64;
            for (&w, &cv) in row.iter().zip(cache.conv_out[..d].iter()) {
                acc += w * cv;
            }
            y_out[r] = acc;
        }
        Ok(())
    }
}

// ===========================================================================
// 3. Selective State-Space Model — Mamba-2 (Granite 4.0 H)
// ===========================================================================

/// Recurrent state for `SelectiveSsmMamba2`: `[n_heads, d_head, d_state]` (bounded O(1) memory).
#[derive(Clone, Debug)]
pub struct Mamba2State {
    pub n_heads: usize,
    pub d_head: usize,
    pub d_state: usize,
    pub h_state: Vec<f64>,
}

impl Mamba2State {
    pub fn new(n_heads: usize, d_head: usize, d_state: usize) -> Self {
        Self {
            n_heads,
            d_head,
            d_state,
            h_state: vec![0.0_f64; n_heads * d_head * d_state],
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.h_state.fill(0.0);
    }
}

/// Granite 4.0 H Mamba-2 selective state-space step parameters (scalar per head):
///   `dt_h    = softplus(dt_raw_h + dt_bias_h)`
///   `decay_h = exp(-dt_h * exp(A_log_h))`
///   `H_t[h, p, n] = decay_h * H_{t-1}[h, p, n] + dt_h * B[n] * x[h, p]`
///   `y_t[h, p]    = sum_n C[n] * H_t[h, p, n] + D[h] * x[h, p]`
#[derive(Clone, Debug)]
pub struct SelectiveSsmMamba2 {
    pub n_heads: usize,
    pub d_head: usize,
    pub d_state: usize,
    pub a_log: Vec<f64>,   // [n_heads]
    pub d_skip: Vec<f64>,  // [n_heads]
    pub dt_bias: Vec<f64>, // [n_heads]
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

impl SelectiveSsmMamba2 {
    pub fn new(
        n_heads: usize,
        d_head: usize,
        d_state: usize,
        a_log: Vec<f64>,
        d_skip: Vec<f64>,
        dt_bias: Vec<f64>,
    ) -> Result<Self, ErrorCode> {
        if n_heads == 0
            || d_head == 0
            || d_state == 0
            || a_log.len() != n_heads
            || d_skip.len() != n_heads
            || dt_bias.len() != n_heads
        {
            return Err(ErrorCode::ShapeMismatch);
        }
        Ok(Self {
            n_heads,
            d_head,
            d_state,
            a_log,
            d_skip,
            dt_bias,
        })
    }

    /// Zero-allocation recurrent Mamba-2 state update for one timestep.
    pub fn step_into(
        &self,
        x_heads: &[f64], // [n_heads * d_head]
        b_vec: &[f64],   // [d_state]
        c_vec: &[f64],   // [d_state]
        dt_raw: &[f64],  // [n_heads]
        state: &mut Mamba2State,
        y_out: &mut [f64], // [n_heads * d_head]
    ) -> Result<(), ErrorCode> {
        let total_d = self.n_heads * self.d_head;
        if x_heads.len() < total_d
            || b_vec.len() < self.d_state
            || c_vec.len() < self.d_state
            || dt_raw.len() < self.n_heads
            || y_out.len() < total_d
            || state.h_state.len() < total_d * self.d_state
        {
            return Err(ErrorCode::BufferTooSmall);
        }

        let ds = self.d_state;
        for h in 0..self.n_heads {
            let dt = softplus(dt_raw[h] + self.dt_bias[h]);
            let a = -self.a_log[h].exp();
            let decay = (dt * a).exp();
            let d_skip = self.d_skip[h];

            for p in 0..self.d_head {
                let hp = h * self.d_head + p;
                let x_val = x_heads[hp];
                let st_off = hp * ds;
                let st_row = &mut state.h_state[st_off..st_off + ds];
                let mut y_acc = d_skip * x_val;
                for n in 0..ds {
                    let new_h = decay * st_row[n] + dt * b_vec[n] * x_val;
                    st_row[n] = new_h;
                    y_acc += c_vec[n] * new_h;
                }
                y_out[hp] = y_acc;
            }
        }
        Ok(())
    }
}

// ===========================================================================
// 4. Dual-Timescale Hierarchical Recurrence Controller (HRM-Text 1B)
// ===========================================================================

/// Configuration for HRM-style nested `H_cycles x L_cycles` recurrent reasoning.
#[derive(Clone, Debug)]
pub struct DualTimescaleConfig {
    pub dim: usize,
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub halt_delta_tol: f64,
    pub z_l_init: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct DualTimescaleTrace {
    pub h_steps_executed: usize,
    pub l_steps_executed: usize,
    pub final_h_delta: f64,
    pub halted_early: bool,
}

/// Preallocated scratch state for `DualTimescaleCycleController`.
#[derive(Clone, Debug)]
pub struct DualTimescaleScratch {
    pub z_h: Vec<f64>,
    pub z_l: Vec<f64>,
    pub inj_buf: Vec<f64>,
    pub next_buf: Vec<f64>,
}

impl DualTimescaleScratch {
    pub fn new(dim: usize) -> Self {
        Self {
            z_h: vec![0.0_f64; dim],
            z_l: vec![0.0_f64; dim],
            inj_buf: vec![0.0_f64; dim],
            next_buf: vec![0.0_f64; dim],
        }
    }
}

/// Executes HRM dual-timescale nested loops with additive state injection:
///   For each `h` in `0..h_cycles`:
///     For each `l` in `0..l_cycles`:
///       `z_L = L_step(z_L + z_H + x_input)`
///     `z_H = H_step(z_H + z_L)`
///   Halts early if `||z_H^{new} - z_H^{old}||_2 <= halt_delta_tol`.
pub fn run_dual_timescale_cycles<FL, FH>(
    cfg: &DualTimescaleConfig,
    x_input: &[f64],
    mut l_step: FL,
    mut h_step: FH,
    scratch: &mut DualTimescaleScratch,
) -> Result<DualTimescaleTrace, ErrorCode>
where
    FL: FnMut(&[f64], &mut [f64]),
    FH: FnMut(&[f64], &mut [f64]),
{
    let d = cfg.dim;
    if d == 0
        || cfg.h_cycles == 0
        || cfg.l_cycles == 0
        || cfg.z_l_init.len() != d
        || x_input.len() < d
        || scratch.z_h.len() < d
    {
        return Err(ErrorCode::BufferTooSmall);
    }

    scratch.z_h[..d].fill(0.0);
    scratch.z_l[..d].copy_from_slice(&cfg.z_l_init);

    let mut h_exec = 0usize;
    let mut l_exec = 0usize;
    let mut final_delta = f64::INFINITY;
    let mut halted_early = false;

    for _hc in 0..cfg.h_cycles {
        for _lc in 0..cfg.l_cycles {
            for i in 0..d {
                scratch.inj_buf[i] = scratch.z_l[i] + scratch.z_h[i] + x_input[i];
            }
            l_step(&scratch.inj_buf[..d], &mut scratch.next_buf[..d]);
            scratch.z_l[..d].copy_from_slice(&scratch.next_buf[..d]);
            l_exec += 1;
        }

        for i in 0..d {
            scratch.inj_buf[i] = scratch.z_h[i] + scratch.z_l[i];
        }
        h_step(&scratch.inj_buf[..d], &mut scratch.next_buf[..d]);

        let mut diff_sq = 0.0_f64;
        for i in 0..d {
            let delta = scratch.next_buf[i] - scratch.z_h[i];
            diff_sq += delta * delta;
            scratch.z_h[i] = scratch.next_buf[i];
        }
        final_delta = diff_sq.sqrt();
        h_exec += 1;

        if cfg.halt_delta_tol > 0.0 && final_delta <= cfg.halt_delta_tol {
            halted_early = true;
            break;
        }
    }

    Ok(DualTimescaleTrace {
        h_steps_executed: h_exec,
        l_steps_executed: l_exec,
        final_h_delta: final_delta,
        halted_early,
    })
}

// ===========================================================================
// 5. Parameterless RMSNorm & Gated Attention GQKV (HRM-Text 1B)
// ===========================================================================

/// Parameterless RMSNorm (`HRM-Text 1B`): zero learned weight parameters, zero heap allocation.
///   `out[i] = x[i] / sqrt(mean(x^2) + eps)`
#[inline]
pub fn parameterless_rmsnorm_into(x: &[f64], eps: f64, out: &mut [f64]) -> Result<(), ErrorCode> {
    let d = x.len();
    if d == 0 || out.len() < d {
        return Err(ErrorCode::BufferTooSmall);
    }
    let mut sq = 0.0_f64;
    for &v in x {
        sq += v * v;
    }
    let inv = 1.0 / ((sq / (d as f64)) + eps).sqrt();
    for (dst, &src) in out[..d].iter_mut().zip(x.iter()) {
        *dst = src * inv;
    }
    Ok(())
}

/// Preallocated scratch for `GatedAttentionGqkv` (single-sequence causal attention).
#[derive(Clone, Debug)]
pub struct GqkvScratch {
    pub dim: usize,
    pub max_seq: usize,
    pub g_buf: Vec<f64>,
    pub q_buf: Vec<f64>,
    pub k_buf: Vec<f64>,
    pub v_buf: Vec<f64>,
    pub attn_out: Vec<f64>,
    pub scores: Vec<f64>,
}

impl GqkvScratch {
    pub fn new(dim: usize, max_seq: usize) -> Self {
        let ms = max_seq.max(1);
        Self {
            dim,
            max_seq: ms,
            g_buf: vec![0.0_f64; ms * dim],
            q_buf: vec![0.0_f64; ms * dim],
            k_buf: vec![0.0_f64; ms * dim],
            v_buf: vec![0.0_f64; ms * dim],
            attn_out: vec![0.0_f64; ms * dim],
            scores: vec![0.0_f64; ms],
        }
    }
}

/// HRM-Text 1B Gated Multi-Head Attention (`gqkv_proj [4*d, d]` + `o_proj [d, d]`):
///   `[G, Q, K, V] = x @ gqkv_proj^T`
///   `A = CausalAttention(Q, K, V)`
///   `out = (sigmoid(G) * A) @ o_proj^T`
#[derive(Clone, Debug)]
pub struct GatedAttentionGqkv {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Row-major `[4 * dim, dim]` (`G`, `Q`, `K`, `V`)
    pub gqkv_proj: Vec<f64>,
    /// Row-major `[dim, dim]`
    pub o_proj: Vec<f64>,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl GatedAttentionGqkv {
    pub fn new(
        dim: usize,
        num_heads: usize,
        gqkv_proj: Vec<f64>,
        o_proj: Vec<f64>,
    ) -> Result<Self, ErrorCode> {
        if dim == 0
            || num_heads == 0
            || !dim.is_multiple_of(num_heads)
            || gqkv_proj.len() != 4 * dim * dim
            || o_proj.len() != dim * dim
        {
            return Err(ErrorCode::ShapeMismatch);
        }
        Ok(Self {
            dim,
            num_heads,
            head_dim: dim / num_heads,
            gqkv_proj,
            o_proj,
        })
    }

    /// Zero-allocation causal gated attention over `seq_len` tokens `[seq_len * dim]`.
    pub fn forward_seq_into(
        &self,
        x_seq: &[f64],
        seq_len: usize,
        scratch: &mut GqkvScratch,
        y_out: &mut [f64],
    ) -> Result<(), ErrorCode> {
        let d = self.dim;
        if seq_len == 0
            || seq_len > scratch.max_seq
            || scratch.dim < d
            || x_seq.len() < seq_len * d
            || y_out.len() < seq_len * d
        {
            return Err(ErrorCode::BufferTooSmall);
        }

        // 1. Project G, Q, K, V
        for t in 0..seq_len {
            let xt = &x_seq[t * d..(t + 1) * d];
            for (blk, dst_all) in [
                (0usize, &mut scratch.g_buf),
                (1usize, &mut scratch.q_buf),
                (2usize, &mut scratch.k_buf),
                (3usize, &mut scratch.v_buf),
            ] {
                let dst = &mut dst_all[t * d..(t + 1) * d];
                for r in 0..d {
                    let row = &self.gqkv_proj[(blk * d + r) * d..(blk * d + r + 1) * d];
                    let mut acc = 0.0_f64;
                    for (&w, &xv) in row.iter().zip(xt.iter()) {
                        acc += w * xv;
                    }
                    dst[r] = acc;
                }
            }
        }

        // 2. Causal multi-head attention + per-element sigmoid(G) output gate
        let dh = self.head_dim;
        let scale = 1.0 / (dh as f64).sqrt();
        scratch.attn_out[..seq_len * d].fill(0.0);

        for h in 0..self.num_heads {
            let hoff = h * dh;
            for t in 0..seq_len {
                let qt = &scratch.q_buf[t * d + hoff..t * d + hoff + dh];
                let mut max_s = f64::NEG_INFINITY;
                for s in 0..=t {
                    let ks = &scratch.k_buf[s * d + hoff..s * d + hoff + dh];
                    let mut dot = 0.0_f64;
                    for (&qv, &kv) in qt.iter().zip(ks.iter()) {
                        dot += qv * kv;
                    }
                    let sc = dot * scale;
                    scratch.scores[s] = sc;
                    if sc > max_s {
                        max_s = sc;
                    }
                }
                let mut sum_e = 0.0_f64;
                for s in 0..=t {
                    let w = (scratch.scores[s] - max_s).exp();
                    scratch.scores[s] = w;
                    sum_e += w;
                }
                let inv_e = if sum_e > 0.0 { 1.0 / sum_e } else { 0.0 };
                let out_t = &mut scratch.attn_out[t * d + hoff..t * d + hoff + dh];
                for s in 0..=t {
                    let w = scratch.scores[s] * inv_e;
                    let vs = &scratch.v_buf[s * d + hoff..s * d + hoff + dh];
                    for c in 0..dh {
                        out_t[c] += w * vs[c];
                    }
                }
                // Apply HRM sigmoid(G) gate before o_proj
                let gt = &scratch.g_buf[t * d + hoff..t * d + hoff + dh];
                for c in 0..dh {
                    out_t[c] *= sigmoid(gt[c]);
                }
            }
        }

        // 3. o_proj: [d, d]
        for t in 0..seq_len {
            let at = &scratch.attn_out[t * d..(t + 1) * d];
            let yt = &mut y_out[t * d..(t + 1) * d];
            for r in 0..d {
                let row = &self.o_proj[r * d..(r + 1) * d];
                let mut acc = 0.0_f64;
                for (&w, &av) in row.iter().zip(at.iter()) {
                    acc += w * av;
                }
                yt[r] = acc;
            }
        }
        Ok(())
    }
}

// ===========================================================================
// 6. PonderNet Geometric Refinement & Verification Gate (ADA-o1)
// ===========================================================================

/// Preallocated scratch for zero-allocation `PonderNetRefinement::refine_into`.
#[derive(Clone, Debug)]
pub struct PonderRefinementScratch {
    pub ctx_t: Vec<f64>,
    pub ws_prev: Vec<f64>,
    pub ws_curr: Vec<f64>,
}

impl PonderRefinementScratch {
    pub fn new(dim: usize) -> Self {
        Self {
            ctx_t: vec![0.0_f64; dim],
            ws_prev: vec![0.0_f64; dim],
            ws_curr: vec![0.0_f64; dim],
        }
    }
}

/// Telemetry trace returned by `PonderNetRefinement::refine_into`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PonderRefinementTrace {
    pub steps_executed: u32,
    pub halted_early: bool,
    pub final_halt_prob: f64,
    pub final_verify_prob: f64,
}

/// Recurrent PonderNet geometric halting + verification-gated workspace refinement:
///   `ctx_t = ctx_base + w_feedback @ ws_{t-1}`
///   `ws_t  = step_fn(ctx_t)`
///   `lambda_t = sigmoid(w_halt · ws_t + b_halt)`
///   `v_t      = sigmoid(w_verify · ws_t + b_verify)`
///   `ws_acc  += p_t * ws_t * (0.5 * v_t + 0.5)`  where `p_t = r_t * lambda_t`
#[derive(Clone, Debug)]
pub struct PonderNetRefinement {
    pub dim: usize,
    pub t_cap: usize,
    pub lambda_prior: f64,
    pub w_feedback: Vec<f64>, // [dim, dim]
    pub w_halt: Vec<f64>,     // [dim]
    pub b_halt: f64,
    pub w_verify: Vec<f64>, // [dim]
    pub b_verify: f64,
}

impl PonderNetRefinement {
    /// Zero-allocation PonderNet geometric refinement loop writing accumulated state into `ws_acc_out`.
    pub fn refine_into<F>(
        &self,
        ctx_base: &[f64],
        mut step_fn: F,
        scratch: &mut PonderRefinementScratch,
        ws_acc_out: &mut [f64],
    ) -> Result<PonderRefinementTrace, ErrorCode>
    where
        F: FnMut(&[f64], &mut [f64]),
    {
        let d = self.dim;
        if d == 0
            || self.t_cap == 0
            || ctx_base.len() < d
            || ws_acc_out.len() < d
            || scratch.ctx_t.len() < d
            || scratch.ws_prev.len() < d
            || scratch.ws_curr.len() < d
        {
            return Err(ErrorCode::BufferTooSmall);
        }

        scratch.ws_prev[..d].fill(0.0);
        ws_acc_out[..d].fill(0.0);
        let mut r_rem = 1.0_f64;
        let mut steps_executed = 0u32;
        let mut halted_early = false;
        let mut final_halt_prob = 0.0_f64;
        let mut final_verify_prob = 0.0_f64;

        for t in 0..self.t_cap {
            // ctx_t = ctx_base + w_feedback @ ws_prev
            for r in 0..d {
                let row = &self.w_feedback[r * d..(r + 1) * d];
                let mut fb = 0.0_f64;
                for (&w, &wp) in row.iter().zip(scratch.ws_prev[..d].iter()) {
                    fb += w * wp;
                }
                scratch.ctx_t[r] = ctx_base[r] + fb;
            }

            step_fn(&scratch.ctx_t[..d], &mut scratch.ws_curr[..d]);

            let mut h_logit = self.b_halt;
            let mut v_logit = self.b_verify;
            for i in 0..d {
                let wv = scratch.ws_curr[i];
                h_logit += self.w_halt[i] * wv;
                v_logit += self.w_verify[i] * wv;
            }
            let lam = sigmoid(h_logit);
            let ver = sigmoid(v_logit);
            final_halt_prob = lam;
            final_verify_prob = ver;
            steps_executed += 1;

            let is_last = t + 1 == self.t_cap;
            let should_halt = lam >= 0.5 && ver >= 0.5;
            let p_t = if is_last || should_halt {
                r_rem
            } else {
                r_rem * lam
            };
            let gate = 0.5 * ver + 0.5;
            for i in 0..d {
                ws_acc_out[i] += p_t * scratch.ws_curr[i] * gate;
            }
            scratch.ws_prev[..d].copy_from_slice(&scratch.ws_curr[..d]);
            r_rem *= 1.0 - lam;

            if should_halt {
                halted_early = !is_last;
                break;
            }
        }

        Ok(PonderRefinementTrace {
            steps_executed,
            halted_early,
            final_halt_prob,
            final_verify_prob,
        })
    }
}
