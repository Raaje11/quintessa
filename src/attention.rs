//! Attention Lowering & Reference Kernels (KDA + MLA, dynamic layer topology).
//!
//! Zero runtime heap allocation: all state matrices and latent caches live in
//! `AttentionScratch`.

use crate::ir::AttentionIr;
use crate::sys::{AttentionKind, ErrorCode};

pub struct AttentionScratch {
    pub substrate_dim: usize,
    pub state_dim: usize,
    pub kv_latent_dim: usize,
    /// KDA recurrent state per KDA layer: `[num_kda_layers * state_dim * state_dim]`
    pub kda_state: Vec<f64>,
    /// MLA latent KV cache per MLA layer: `[num_mla_layers * max_seq_len * kv_latent_dim]`
    pub mla_latent_cache: Vec<f64>,
    pub k_buf: Vec<f64>,
    pub v_buf: Vec<f64>,
    pub q_buf: Vec<f64>,
    pub scores_buf: Vec<f64>,
    pub cur_buf: Vec<f64>,
    pub max_seq_len: usize,
}

impl AttentionScratch {
    pub fn new(ir: &AttentionIr, substrate_dim: usize, max_seq_len: usize) -> Self {
        let mut num_kda = 0usize;
        let mut num_mla = 0usize;
        let mut state_dim = 0usize;
        let mut kv_latent_dim = 0usize;
        for layer in &ir.layers {
            match layer.kind {
                AttentionKind::Kda => {
                    num_kda += 1;
                    state_dim = state_dim.max((layer.kda.state_dim as usize).min(substrate_dim));
                }
                AttentionKind::Mla => {
                    num_mla += 1;
                    kv_latent_dim =
                        kv_latent_dim.max((layer.mla.kv_latent_dim as usize).min(substrate_dim));
                }
            }
        }
        Self {
            substrate_dim,
            state_dim,
            kv_latent_dim,
            kda_state: vec![0.0; num_kda * state_dim * state_dim],
            mla_latent_cache: vec![0.0; num_mla * max_seq_len * kv_latent_dim],
            k_buf: vec![0.0; state_dim],
            v_buf: vec![0.0; state_dim],
            q_buf: vec![0.0; state_dim],
            scores_buf: vec![0.0; max_seq_len],
            cur_buf: vec![0.0; substrate_dim],
            max_seq_len,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.kda_state.fill(0.0);
        self.mla_latent_cache.fill(0.0);
    }
}

/// Reference KDA delta-rule recurrent step for one token/vector `x[..dim]`:
///   S_t = alpha * (S_{t-1} - (S_{t-1} k) k^T) + v k^T
///   y   = S_t q
pub fn reference_kda_step(
    kda_idx: usize,
    alpha: f64,
    x: &[f64],
    out: &mut [f64],
    scratch: &mut AttentionScratch,
) -> Result<(), ErrorCode> {
    let sd = scratch.state_dim;
    if kda_idx >= 6 || x.len() < sd || out.len() < sd {
        return Err(ErrorCode::BufferTooSmall);
    }

    // Normalize k for stable delta-rule update
    let mut k_norm_sq = 0.0;
    for i in 0..sd {
        let kv = x[i];
        scratch.k_buf[i] = kv;
        k_norm_sq += kv * kv;
    }
    let inv_norm = if k_norm_sq > 1e-12 {
        1.0 / k_norm_sq.sqrt()
    } else {
        0.0
    };
    for i in 0..sd {
        scratch.k_buf[i] *= inv_norm;
        scratch.v_buf[i] = x[i];
        scratch.q_buf[i] = scratch.k_buf[i];
    }

    let state_off = kda_idx * sd * sd;
    let state = &mut scratch.kda_state[state_off..state_off + sd * sd];

    // Update S_t and read y = S_t q
    out.fill(0.0);
    for r in 0..sd {
        let row_off = r * sd;
        // s_k = dot(S[r, :], k)
        let mut s_k = 0.0;
        for c in 0..sd {
            s_k += state[row_off + c] * scratch.k_buf[c];
        }
        let vr = scratch.v_buf[r];
        let mut y_r = 0.0;
        for c in 0..sd {
            let kc = scratch.k_buf[c];
            let new_s = alpha * (state[row_off + c] - s_k * kc) + vr * kc;
            state[row_off + c] = new_s;
            y_r += new_s * scratch.q_buf[c];
        }
        out[r] = y_r;
    }

    Ok(())
}

/// Reference MLA latent KV compression + attention step at sequence position `pos`.
pub fn reference_mla_step(
    mla_idx: usize,
    pos: usize,
    x: &[f64],
    out: &mut [f64],
    scratch: &mut AttentionScratch,
) -> Result<(), ErrorCode> {
    let ld = scratch.kv_latent_dim;
    if mla_idx >= 2 || pos >= scratch.max_seq_len || x.len() < ld || out.len() < ld {
        return Err(ErrorCode::BufferTooSmall);
    }

    let cache_base = mla_idx * scratch.max_seq_len * ld;
    // Compress x into latent KV slot `pos`
    let dst = &mut scratch.mla_latent_cache[cache_base + pos * ld..cache_base + (pos + 1) * ld];
    dst.copy_from_slice(&x[..ld]);

    let inv_sqrt_d = 1.0 / (ld as f64).sqrt();
    let mut max_s = f64::NEG_INFINITY;
    for t in 0..=pos {
        let k_slice = &scratch.mla_latent_cache[cache_base + t * ld..cache_base + (t + 1) * ld];
        let mut dot = 0.0;
        for i in 0..ld {
            dot += x[i] * k_slice[i];
        }
        let s = dot * inv_sqrt_d;
        scratch.scores_buf[t] = s;
        if s > max_s {
            max_s = s;
        }
    }

    let mut sum_exp = 0.0;
    for t in 0..=pos {
        let w = (scratch.scores_buf[t] - max_s).exp();
        scratch.scores_buf[t] = w;
        sum_exp += w;
    }
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

    out.fill(0.0);
    for t in 0..=pos {
        let w = scratch.scores_buf[t] * inv_sum;
        let v_slice = &scratch.mla_latent_cache[cache_base + t * ld..cache_base + (t + 1) * ld];
        for i in 0..ld {
            out[i] += w * v_slice[i];
        }
    }

    Ok(())
}

/// Dispatch one pass through the 8-layer `[KDA, KDA, KDA, MLA, KDA, KDA, KDA, MLA]` stack.
pub fn run_attention_stack(
    ir: &AttentionIr,
    pos: usize,
    x: &[f64],
    out: &mut [f64],
    scratch: &mut AttentionScratch,
) -> Result<(), ErrorCode> {
    if x.len() > scratch.cur_buf.len() || out.len() < x.len() {
        return Err(ErrorCode::BufferTooSmall);
    }
    let dim = x.len();
    let mut kda_idx = 0usize;
    let mut mla_idx = 0usize;
    scratch.cur_buf[..dim].copy_from_slice(x);

    for layer in &ir.layers {
        match layer.kind {
            AttentionKind::Kda => {
                // Temporarily copy cur_buf to out? No, reference_kda_step reads x and writes out.
                // Split borrow: use a raw slice or copy via scratch.scores_buf? No, cur_buf is in scratch!
                // Simply copy cur_buf into out, then read from out? No, reference_kda_step writes out!
                // Even cleaner: add `layer_in_buf` or use `cur_buf` in `run_attention_stack` by passing
                // `out` as scratchpad? Let's pass `cur_buf` separately by splitting AttentionScratch or
                // copying `x` into `out` first? Wait: `reference_kda_step` only needs `k_buf, v_buf, q_buf, kda_state`.
                // So we can pass `cur_buf` if we use `split_cur` or a small inline helper.
                reference_kda_step_internal(kda_idx, 0.9, dim, out, scratch)?;
                kda_idx += 1;
            }
            AttentionKind::Mla => {
                reference_mla_step_internal(mla_idx, pos, dim, out, scratch)?;
                mla_idx += 1;
            }
        }
        for i in 0..dim {
            scratch.cur_buf[i] += out[i];
        }
    }
    out[..dim].copy_from_slice(&scratch.cur_buf[..dim]);
    Ok(())
}

fn reference_kda_step_internal(
    kda_idx: usize,
    alpha: f64,
    dim: usize,
    out: &mut [f64],
    scratch: &mut AttentionScratch,
) -> Result<(), ErrorCode> {
    let sd = scratch.state_dim;
    if kda_idx >= 6 || dim < sd || out.len() < sd {
        return Err(ErrorCode::BufferTooSmall);
    }
    let mut k_norm_sq = 0.0;
    for i in 0..sd {
        let kv = scratch.cur_buf[i];
        scratch.k_buf[i] = kv;
        k_norm_sq += kv * kv;
    }
    let inv_norm = if k_norm_sq > 1e-12 {
        1.0 / k_norm_sq.sqrt()
    } else {
        0.0
    };
    for i in 0..sd {
        scratch.k_buf[i] *= inv_norm;
        scratch.v_buf[i] = scratch.cur_buf[i];
        scratch.q_buf[i] = scratch.k_buf[i];
    }

    let state_off = kda_idx * sd * sd;
    let state = &mut scratch.kda_state[state_off..state_off + sd * sd];
    out.fill(0.0);
    for r in 0..sd {
        let row_off = r * sd;
        let mut s_k = 0.0;
        for c in 0..sd {
            s_k += state[row_off + c] * scratch.k_buf[c];
        }
        let vr = scratch.v_buf[r];
        let mut y_r = 0.0;
        for c in 0..sd {
            let kc = scratch.k_buf[c];
            let new_s = alpha * (state[row_off + c] - s_k * kc) + vr * kc;
            state[row_off + c] = new_s;
            y_r += new_s * scratch.q_buf[c];
        }
        out[r] = y_r;
    }
    Ok(())
}

fn reference_mla_step_internal(
    mla_idx: usize,
    pos: usize,
    dim: usize,
    out: &mut [f64],
    scratch: &mut AttentionScratch,
) -> Result<(), ErrorCode> {
    let ld = scratch.kv_latent_dim;
    if mla_idx >= 2 || pos >= scratch.max_seq_len || dim < ld || out.len() < ld {
        return Err(ErrorCode::BufferTooSmall);
    }
    let cache_base = mla_idx * scratch.max_seq_len * ld;
    let dst = &mut scratch.mla_latent_cache[cache_base + pos * ld..cache_base + (pos + 1) * ld];
    dst.copy_from_slice(&scratch.cur_buf[..ld]);

    let inv_sqrt_d = 1.0 / (ld as f64).sqrt();
    let mut max_s = f64::NEG_INFINITY;
    for t in 0..=pos {
        let k_slice = &scratch.mla_latent_cache[cache_base + t * ld..cache_base + (t + 1) * ld];
        let mut dot = 0.0;
        for i in 0..ld {
            dot += scratch.cur_buf[i] * k_slice[i];
        }
        let s = dot * inv_sqrt_d;
        scratch.scores_buf[t] = s;
        if s > max_s {
            max_s = s;
        }
    }

    let mut sum_exp = 0.0;
    for t in 0..=pos {
        let w = (scratch.scores_buf[t] - max_s).exp();
        scratch.scores_buf[t] = w;
        sum_exp += w;
    }
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

    out.fill(0.0);
    for t in 0..=pos {
        let w = scratch.scores_buf[t] * inv_sum;
        let v_slice = &scratch.mla_latent_cache[cache_base + t * ld..cache_base + (t + 1) * ld];
        for i in 0..ld {
            out[i] += w * v_slice[i];
        }
    }
    Ok(())
}

/// Encode a token sequence `[seq_len]` through sinusoidal positional modulation and
/// the compiled `KDA + MLA` attention stack into `out [dim]`, using zero heap allocation.
pub fn encode_token_sequence(
    ir: &AttentionIr,
    tokens: &[u16],
    tok_emb: &[f64],
    vocab_size: usize,
    dim: usize,
    out: &mut [f64],
    scratch: &mut AttentionScratch,
) -> Result<(), ErrorCode> {
    if dim == 0 || dim > 512 || out.len() < dim || scratch.cur_buf.len() < dim {
        return Err(ErrorCode::BufferTooSmall);
    }
    scratch.reset();
    let len = tokens.len().min(scratch.max_seq_len);
    if len == 0 {
        out[..dim].fill(0.0);
        return Ok(());
    }

    let mut tok_buf = [0.0_f64; 512];
    let mut step_out = [0.0_f64; 512];
    let mut mean_acc = [0.0_f64; 512];

    for (pos, &tok) in tokens[..len].iter().enumerate() {
        let tid = tok as usize;
        if tid < vocab_size {
            let src = &tok_emb[tid * dim..(tid + 1) * dim];
            tok_buf[..dim].copy_from_slice(src);
        } else {
            tok_buf[..dim].fill(0.0);
        }
        // Sinusoidal positional modulation
        let p = (pos + 1) as f64;
        for d in 0..dim {
            let freq = 1.0 / (10000.0_f64).powf((2 * (d / 2)) as f64 / (dim as f64));
            let pe = if d % 2 == 0 {
                (p * freq).sin()
            } else {
                (p * freq).cos()
            };
            tok_buf[d] += 0.1 * pe;
        }
        let mut norm_sq = 0.0;
        for &v in &tok_buf[..dim] {
            norm_sq += v * v;
        }
        if norm_sq > 1e-12 {
            let inv = 1.0 / norm_sq.sqrt();
            for v in &mut tok_buf[..dim] {
                *v *= inv;
            }
        }

        run_attention_stack(ir, pos, &tok_buf[..dim], &mut step_out[..dim], scratch)?;
        let mut out_norm_sq = 0.0;
        for &v in &step_out[..dim] {
            out_norm_sq += v * v;
        }
        if out_norm_sq > 1e-12 {
            let inv = 1.0 / out_norm_sq.sqrt();
            for d in 0..dim {
                step_out[d] *= inv;
            }
        }
        for d in 0..dim {
            mean_acc[d] += step_out[d];
        }
    }

    let mut mean_norm_sq = 0.0;
    for &v in &mean_acc[..dim] {
        mean_norm_sq += v * v;
    }
    let inv_mean = if mean_norm_sq > 1e-12 {
        1.0 / mean_norm_sq.sqrt()
    } else {
        0.0
    };
    for d in 0..dim {
        out[d] = 0.5 * step_out[d] + 0.5 * (mean_acc[d] * inv_mean);
    }
    let mut final_norm_sq = 0.0;
    for &v in &out[..dim] {
        final_norm_sq += v * v;
    }
    if final_norm_sq > 1e-12 {
        let inv = 1.0 / final_norm_sq.sqrt();
        for v in &mut out[..dim] {
            *v *= inv;
        }
    }
    Ok(())
}

/// Encode a token sequence into `num_slots` positional embedding slots `[num_slots * dim]`
/// with sinusoidal positional modulation and per-slot L2 normalization (zero heap allocation).
pub fn encode_token_slots(
    tokens: &[u16],
    tok_emb: &[f64],
    vocab_size: usize,
    dim: usize,
    num_slots: usize,
    out_slots: &mut [f64],
) -> Result<(), ErrorCode> {
    if dim == 0 || num_slots == 0 || out_slots.len() < num_slots * dim {
        return Err(ErrorCode::BufferTooSmall);
    }
    out_slots[..num_slots * dim].fill(0.0);
    if tokens.is_empty() {
        return Ok(());
    }
    let len = tokens.len();
    for (pos, &tok) in tokens.iter().enumerate() {
        let tid = tok as usize;
        if tid >= vocab_size {
            continue;
        }
        let slot = (pos * num_slots) / len;
        let dst = &mut out_slots[slot * dim..(slot + 1) * dim];
        let src = &tok_emb[tid * dim..(tid + 1) * dim];
        let p = (pos + 1) as f64;
        for d in 0..dim {
            let freq = 1.0 / (10000.0_f64).powf((2 * (d / 2)) as f64 / (dim as f64));
            let pe = if d % 2 == 0 {
                (p * freq).sin()
            } else {
                (p * freq).cos()
            };
            dst[d] += src[d] + 0.05 * pe;
        }
    }
    for s in 0..num_slots {
        let dst = &mut out_slots[s * dim..(s + 1) * dim];
        let mut norm_sq = 0.0;
        for &v in dst.iter() {
            norm_sq += v * v;
        }
        if norm_sq > 1e-12 {
            let inv = 1.0 / norm_sq.sqrt();
            for v in dst.iter_mut() {
                *v *= inv;
            }
        }
    }
    Ok(())
}

/// Write unblurred per-position token embeddings `[max_len, dim]` for `tokens` into `out_mat`
/// (zero-padded beyond `tokens.len()`, zero heap allocation).
pub fn encode_prompt_matrix(
    tokens: &[u16],
    tok_emb: &[f64],
    vocab_size: usize,
    dim: usize,
    max_len: usize,
    out_mat: &mut [f64],
) -> Result<(), ErrorCode> {
    if dim == 0 || max_len == 0 || out_mat.len() < max_len * dim {
        return Err(ErrorCode::BufferTooSmall);
    }
    out_mat[..max_len * dim].fill(0.0);
    let len = tokens.len().min(max_len);
    for (pos, &tok) in tokens[..len].iter().enumerate() {
        let tid = tok as usize;
        if tid < vocab_size {
            out_mat[pos * dim..(pos + 1) * dim]
                .copy_from_slice(&tok_emb[tid * dim..(tid + 1) * dim]);
        }
    }
    Ok(())
}
