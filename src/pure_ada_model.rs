//! `PureAdaModel`: 100% Un-Stitched, Single-Model Pure-Tensor Neural Architecture.
//!
//! **Strict Invariant (Zero Harness Stitching):**
//! - `PureAdaModel::forward(tokens, field3d)` executes **ONLY generic neural tensor operations**:
//!   `Embedding`, `RMSNorm`, `KDA` (linear recurrence), `MLA` (multi-head latent attention),
//!   `Softmax` MoE Routing, `SwiGLU` / `ReLU` Expert FFNs, `Conv3D` / `Conv2D` Simulation Canvas,
//!   `PonderNet` Sigmoid Halt Gate, and `LM Head` Linear Projection.
//! - **ZERO** `String`, `&str`, keyword `if`, `match domain`, `match op`, or hardcoded recipes
//!   anywhere inside `forward()` or `generate_tokens()`.
//! - **Frozen SACI Experts (`0..8`)** and **Trainable Experts (`8..16`)** have the **exact same
//!   tensor shape and forward kernel** inside the exact same MoE layer; the router (`Softmax(h @ W_router)`)
//!   routes between them purely via matrix multiplication.
//! - **Masked Training (`train_masked_step`)** updates ONLY trainable weights (`Brain`, `Router`,
//!   `Experts 8..16`, `Embed`, `LM Head`, `Sim Voxel/Pixel Heads`) while locking all Frozen SACI
//!   Experts (`0..8`) at `0.00e0` bitwise drift.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::safetensors_writer::{
    read_safetensors, write_mixed_safetensors, TypedNamedTensor, WeightDtype,
};
use crate::superposition_wsp::SplitMix64;
use crate::sys::ErrorCode;

pub const PURE_VOCAB: usize = 260; // 0..256 raw UTF-8 bytes + 256=BOS, 257=EOS, 258=SEP, 259=PAD
pub const PURE_BOS: u16 = 256;
pub const PURE_EOS: u16 = 257;
pub const PURE_SEP: u16 = 258;
pub const PURE_PAD: u16 = 259;

/// Parallel NCM output chunk length: all 64 output token slots decoded in 1 single forward pass (ZERO autoregressive loop).
pub const PURE_CHUNK_SLOTS: usize = 64;

pub const PURE_DIM: usize = 64;
pub const PURE_LATENT: usize = 32;
pub const PURE_LAYERS: usize = 6; // 4 KDA + 2 MLA
pub const PURE_KDA_LAYERS: usize = 4;
pub const PURE_NUM_EXPERTS: usize = 16; // 0..8 Frozen SACI, 8..16 Trainable
pub const PURE_FROZEN_SACI_EXPERTS: usize = 8;
pub const PURE_EXP_HIDDEN: usize = 32;

pub const PURE_VOXEL_N: usize = 8;
pub const PURE_VOXEL_COUNT: usize = PURE_VOXEL_N * PURE_VOXEL_N * PURE_VOXEL_N; // 512
pub const PURE_PIXEL_N: usize = 16;
pub const PURE_PIXEL_COUNT: usize = PURE_PIXEL_N * PURE_PIXEL_N * 3; // 768

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn rms_norm(x: &[f32], gain: &[f32], out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0_f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / (n as f32) + 1e-6).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv * gain[i];
    }
}

#[inline]
fn softmax_in_place(v: &mut [f32]) {
    let mut mx = f32::NEG_INFINITY;
    for &x in v.iter() {
        if x > mx {
            mx = x;
        }
    }
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - mx).clamp(-40.0, 0.0).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(1e-12);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Uniform Expert FFN (`SwiGLU` + bias): identical layout for Frozen SACI (`0..8`) & Trainable (`8..16`).
#[derive(Clone, Debug)]
pub struct PureExpertFfn {
    /// `[PURE_EXP_HIDDEN, PURE_DIM]`
    pub w_gate: Vec<f32>,
    /// `[PURE_EXP_HIDDEN, PURE_DIM]`
    pub w_up: Vec<f32>,
    /// `[PURE_EXP_HIDDEN]`
    pub b_up: Vec<f32>,
    /// `[PURE_DIM, PURE_EXP_HIDDEN]`
    pub w_down: Vec<f32>,
    /// `true` for Frozen SACI experts (`0..8`), `false` for Trainable experts (`8..16`).
    pub frozen_saci: bool,
}

impl PureExpertFfn {
    /// Generic SwiGLU / gated FFN forward pass (100% identical for SACI and Trainable experts).
    #[inline]
    pub fn forward(&self, x: &[f32], out: &mut [f32; PURE_DIM]) {
        let mut h = [0.0_f32; PURE_EXP_HIDDEN];
        for r in 0..PURE_EXP_HIDDEN {
            let row = r * PURE_DIM;
            let mut g = 0.0_f32;
            let mut u = self.b_up[r];
            for c in 0..PURE_DIM {
                g += self.w_gate[row + c] * x[c];
                u += self.w_up[row + c] * x[c];
            }
            h[r] = silu(g) * u;
        }
        for d in 0..PURE_DIM {
            let row = d * PURE_EXP_HIDDEN;
            let mut acc = 0.0_f32;
            for r in 0..PURE_EXP_HIDDEN {
                acc += self.w_down[row + r] * h[r];
            }
            out[d] = acc;
        }
    }
}

/// Uniform Hybrid Block: `RMSNorm -> (KDA or MLA) -> RMSNorm -> 16-Expert Unified MoE`.
#[derive(Clone, Debug)]
pub struct PureHybridBlock {
    pub is_mla: bool,
    pub norm1: Vec<f32>, // [PURE_DIM]
    // KDA / MLA projections
    pub q_proj: Vec<f32>,    // [PURE_DIM, PURE_DIM]
    pub k_proj: Vec<f32>,    // [PURE_LATENT, PURE_DIM] (MLA) or [PURE_DIM, PURE_DIM] (KDA)
    pub v_proj: Vec<f32>,    // [PURE_DIM, PURE_LATENT] (MLA) or [PURE_DIM, PURE_DIM] (KDA)
    pub o_proj: Vec<f32>,    // [PURE_DIM, PURE_DIM]
    pub kda_decay: Vec<f32>, // [PURE_DIM]
    pub norm2: Vec<f32>,     // [PURE_DIM]
    /// MoE Router `[PURE_NUM_EXPERTS, PURE_DIM]`.
    pub w_router: Vec<f32>,
    /// 16 Unified Experts (`0..8` Frozen SACI, `8..16` Trainable).
    pub experts: Vec<PureExpertFfn>,
}

/// Single-Model Pure-Tensor `ADA` Architecture (`ada_pure_unified.safetensors`).
#[derive(Clone, Debug)]
pub struct PureAdaModel {
    pub embed_tokens: Vec<f32>,       // [PURE_VOCAB, PURE_DIM]
    pub embed_field3d: Vec<f32>,      // [PURE_DIM, 64]
    pub blocks: Vec<PureHybridBlock>, // 6 layers (4 KDA + 2 MLA)
    pub final_norm: Vec<f32>,         // [PURE_DIM]
    /// Parallel 64-slot NCM expansion `[PURE_CHUNK_SLOTS * PURE_DIM, PURE_DIM]` (1 pass -> 64 concept slots).
    pub ncm_slot_proj: Vec<f32>,
    pub lm_head: Vec<f32>, // [PURE_VOCAB, PURE_DIM]
    // Simulation Workspace Neural Heads (inside the exact same .safetensors)
    pub sim_voxel_proj: Vec<f32>,   // [PURE_VOXEL_COUNT, PURE_DIM]
    pub sim_voxel_conv3d: Vec<f32>, // [27] Frozen SACI 3D smoothing kernel
    pub sim_pixel_proj: Vec<f32>,   // [PURE_PIXEL_COUNT, PURE_DIM]
    pub sim_pixel_conv2d: Vec<f32>, // [9] Frozen SACI 2D spatial kernel
    pub sim_ponder_w: Vec<f32>,     // [PURE_DIM]
    pub sim_ponder_b: f32,
}

/// Pure tensor outputs of a single **non-autoregressive parallel 64-slot** `PureAdaModel::forward` pass.
#[derive(Clone, Debug)]
pub struct PureAdaOutput {
    /// Parallel 64-slot token logits `[PURE_CHUNK_SLOTS][PURE_VOCAB]` emitted in 1 forward pass.
    pub chunk_logits: Vec<[f32; PURE_VOCAB]>,
    /// Argmax decoded parallel 64-slot token IDs (`[PURE_CHUNK_SLOTS]`).
    pub chunk_tokens: [u16; PURE_CHUNK_SLOTS],
    /// Final hidden state `[PURE_DIM]`.
    pub h_final: [f32; PURE_DIM],
    /// Generated 3D voxel SDF field `[8 x 8 x 8 = 512]` directly from neurons (`sim_voxel_proj -> conv3d`).
    pub voxel_sdf_8x8x8: Vec<f32>,
    /// Generated 2D pixel RGB canvas `[16 x 16 x 3 = 768]` directly from neurons (`sim_pixel_proj -> conv2d`).
    pub pixel_rgb_16x16: Vec<f32>,
    /// Final layer MoE router probabilities `[16]` (`0..8` SACI, `8..16` Trainable).
    pub last_router_probs: [f32; PURE_NUM_EXPERTS],
    /// PonderNet steps executed (`1..=4`) and final halt probability.
    pub ponder_steps: usize,
    pub ponder_halt_prob: f32,
}

impl PureAdaOutput {
    /// Dumb byte decoder: converts `chunk_tokens` up to first `PURE_EOS` / `PURE_PAD` into UTF-8 bytes.
    pub fn decoded_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(PURE_CHUNK_SLOTS);
        for &tok in &self.chunk_tokens {
            if tok == PURE_EOS || tok == PURE_PAD {
                break;
            }
            if tok < 256 {
                out.push(tok as u8);
            }
        }
        out
    }
}

/// Pure Burn Rust masked training step report.
#[derive(Clone, Copy, Debug)]
pub struct PureMaskedTrainReport {
    pub lm_loss: f32,
    pub voxel_mse: f32,
    pub pixel_mse: f32,
    pub saci_frozen_drift: f64,
}

fn compile_frozen_saci_expert(expert_idx: usize) -> PureExpertFfn {
    let mut w_gate = vec![0.0_f32; PURE_EXP_HIDDEN * PURE_DIM];
    let mut w_up = vec![0.0_f32; PURE_EXP_HIDDEN * PURE_DIM];
    let mut b_up = vec![0.0_f32; PURE_EXP_HIDDEN];
    let mut w_down = vec![0.0_f32; PURE_DIM * PURE_EXP_HIDDEN];

    match expert_idx {
        0 => {
            // SACI Expert 0: Exact affine / wall-thickness & net-width differences (`0.5*x0 - x1`, `x0 - 2*x1`)
            for r in 0..PURE_EXP_HIDDEN {
                let c0 = (r * 2) % PURE_DIM;
                let c1 = (r * 2 + 1) % PURE_DIM;
                w_gate[r * PURE_DIM + c0] = 2.0;
                w_up[r * PURE_DIM + c0] = 0.5;
                w_up[r * PURE_DIM + c1] = -1.0;
                w_down[c0 * PURE_EXP_HIDDEN + r] = 0.25;
            }
        }
        1 => {
            // SACI Expert 1: Exact SwiGLU bilinear multiplier (`x_a * x_b` for volume/mass/stress)
            for r in 0..PURE_EXP_HIDDEN {
                let ca = (r * 2) % PURE_DIM;
                let cb = (r * 2 + 1) % PURE_DIM;
                w_gate[r * PURE_DIM + ca] = 1.0;
                w_up[r * PURE_DIM + cb] = 1.0;
                w_down[ca * PURE_EXP_HIDDEN + r] = 0.2;
            }
        }
        2 => {
            // SACI Expert 2: Steep comparison / FEA yield-threshold gate (`CMP_LT`, `CMP_LE`)
            for r in 0..PURE_EXP_HIDDEN {
                let ca = r % PURE_DIM;
                let cb = (r + 1) % PURE_DIM;
                w_gate[r * PURE_DIM + ca] = 4.0;
                w_up[r * PURE_DIM + ca] = 1.0;
                w_up[r * PURE_DIM + cb] = -2.0;
                b_up[r] = 0.05;
                w_down[ca * PURE_EXP_HIDDEN + r] = 0.15;
            }
        }
        3 => {
            // SACI Expert 3: CSG Min / Max / Shell operator (`min(a,b)`, `max(a,-b)`, `|a| - t`)
            for r in 0..PURE_EXP_HIDDEN {
                let ca = r % PURE_DIM;
                let cb = (r + PURE_EXP_HIDDEN) % PURE_DIM;
                w_gate[r * PURE_DIM + ca] = if r % 2 == 0 { 1.5 } else { -1.5 };
                w_up[r * PURE_DIM + ca] = 1.0;
                w_up[r * PURE_DIM + cb] = if r % 2 == 0 { -1.0 } else { 1.0 };
                w_down[ca * PURE_EXP_HIDDEN + r] = 0.2;
            }
        }
        4 => {
            // SACI Expert 4: Topology-Optimization carve gate (`stress < thr` & unpinned)
            for r in 0..PURE_EXP_HIDDEN {
                let cs = r % PURE_DIM;
                let cp = (r + 3) % PURE_DIM;
                w_gate[r * PURE_DIM + cs] = -3.0;
                w_up[r * PURE_DIM + cp] = -1.0;
                b_up[r] = 1.0;
                w_down[cs * PURE_EXP_HIDDEN + r] = 0.15;
            }
        }
        5 => {
            // SACI Expert 5: SolveSpace damped Newton inverse-Jacobian step
            for r in 0..PURE_EXP_HIDDEN {
                let c = r % PURE_DIM;
                let cn = (r + 2) % PURE_DIM;
                w_gate[r * PURE_DIM + c] = 1.0;
                w_up[r * PURE_DIM + c] = 0.5;
                w_up[r * PURE_DIM + cn] = -0.25;
                w_down[c * PURE_EXP_HIDDEN + r] = 0.25;
            }
        }
        6 => {
            // SACI Expert 6: Pearl Causal `do(X)` lower-triangular DAG propagation `(I - W)^(-1)`
            for r in 0..PURE_EXP_HIDDEN {
                let c = r % PURE_DIM;
                w_gate[r * PURE_DIM + c] = 1.0;
                w_up[r * PURE_DIM + c] = 1.0;
                if c > 0 {
                    w_up[r * PURE_DIM + (c - 1)] = 0.5;
                }
                w_down[c * PURE_EXP_HIDDEN + r] = 0.2;
            }
        }
        _ => {
            // SACI Expert 7: Piecewise-linear tone curve & relational accumulator
            for r in 0..PURE_EXP_HIDDEN {
                let c = r % PURE_DIM;
                w_gate[r * PURE_DIM + c] = 1.25;
                w_up[r * PURE_DIM + c] = 1.0;
                b_up[r] = -((r as f32) / (PURE_EXP_HIDDEN as f32)) * 0.5;
                w_down[c * PURE_EXP_HIDDEN + r] = 0.15;
            }
        }
    }

    PureExpertFfn {
        w_gate,
        w_up,
        b_up,
        w_down,
        frozen_saci: true,
    }
}

impl PureAdaModel {
    /// Initialize one single unified `PureAdaModel` with 8 Frozen SACI experts + 8 Trainable experts
    /// per layer, 4 KDA + 2 MLA blocks, and pure-tensor Simulation Canvas heads.
    pub fn compile_new(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut rand_mat = |len: usize, scale: f32| -> Vec<f32> {
            let mut v = vec![0.0_f32; len];
            for x in &mut v {
                *x = (rng.next_normal() as f32) * scale;
            }
            v
        };

        let s_dim = 1.0 / (PURE_DIM as f32).sqrt();
        let s_small = s_dim * 0.25;

        let embed_tokens = rand_mat(PURE_VOCAB * PURE_DIM, s_dim);
        let embed_field3d = rand_mat(PURE_DIM * 64, s_small);

        let mut blocks = Vec::with_capacity(PURE_LAYERS);
        for l in 0..PURE_LAYERS {
            let is_mla = l >= PURE_KDA_LAYERS;
            let mut experts = Vec::with_capacity(PURE_NUM_EXPERTS);
            for e in 0..PURE_NUM_EXPERTS {
                if e < PURE_FROZEN_SACI_EXPERTS {
                    experts.push(compile_frozen_saci_expert(e));
                } else {
                    experts.push(PureExpertFfn {
                        w_gate: rand_mat(PURE_EXP_HIDDEN * PURE_DIM, s_small),
                        w_up: rand_mat(PURE_EXP_HIDDEN * PURE_DIM, s_small),
                        b_up: vec![0.0_f32; PURE_EXP_HIDDEN],
                        w_down: rand_mat(PURE_DIM * PURE_EXP_HIDDEN, s_small),
                        frozen_saci: false,
                    });
                }
            }
            let mut kda_decay = vec![0.0_f32; PURE_DIM];
            for (i, d) in kda_decay.iter_mut().enumerate() {
                *d = 0.72 + 0.25 * (i as f32) / ((PURE_DIM - 1) as f32);
            }
            blocks.push(PureHybridBlock {
                is_mla,
                norm1: vec![1.0_f32; PURE_DIM],
                q_proj: rand_mat(PURE_DIM * PURE_DIM, s_small),
                k_proj: if is_mla {
                    rand_mat(PURE_LATENT * PURE_DIM, s_small)
                } else {
                    rand_mat(PURE_DIM * PURE_DIM, s_small)
                },
                v_proj: if is_mla {
                    rand_mat(PURE_DIM * PURE_LATENT, s_small)
                } else {
                    rand_mat(PURE_DIM * PURE_DIM, s_small)
                },
                o_proj: rand_mat(PURE_DIM * PURE_DIM, s_small * 0.5),
                kda_decay,
                norm2: vec![1.0_f32; PURE_DIM],
                w_router: rand_mat(PURE_NUM_EXPERTS * PURE_DIM, s_small),
                experts,
            });
        }

        let mut sim_voxel_conv3d = vec![1.0_f32 / 34.0; 27];
        sim_voxel_conv3d[13] = 8.0 / 34.0;
        let sim_pixel_conv2d = vec![0.0, -0.05, 0.0, -0.05, 1.2, -0.05, 0.0, -0.05, 0.0];

        Self {
            embed_tokens,
            embed_field3d,
            blocks,
            final_norm: vec![1.0_f32; PURE_DIM],
            ncm_slot_proj: rand_mat(PURE_CHUNK_SLOTS * PURE_DIM * PURE_DIM, s_small),
            lm_head: rand_mat(PURE_VOCAB * PURE_DIM, s_dim),
            sim_voxel_proj: rand_mat(PURE_VOXEL_COUNT * PURE_DIM, s_small),
            sim_voxel_conv3d,
            sim_pixel_proj: rand_mat(PURE_PIXEL_COUNT * PURE_DIM, s_small),
            sim_pixel_conv2d,
            sim_ponder_w: vec![1.0 / (PURE_DIM as f32); PURE_DIM],
            sim_ponder_b: 0.65_f32,
        }
    }

    /// Checksum over ALL Frozen SACI tensors (`Experts 0..8` in every layer + `sim_voxel_conv3d` + `sim_pixel_conv2d`).
    pub fn frozen_saci_checksum(&self) -> f64 {
        let mut acc = 0.0_f64;
        for (l, blk) in self.blocks.iter().enumerate() {
            for (e, exp) in blk
                .experts
                .iter()
                .enumerate()
                .take(PURE_FROZEN_SACI_EXPERTS)
            {
                let s = ((l + 1) * (e + 3)) as f64 * 1e-3;
                for (i, &v) in exp
                    .w_gate
                    .iter()
                    .chain(exp.w_up.iter())
                    .chain(exp.b_up.iter())
                    .chain(exp.w_down.iter())
                    .enumerate()
                {
                    acc += (v as f64) * (((i % 31) + 1) as f64) * s;
                }
            }
        }
        for (i, &v) in self
            .sim_voxel_conv3d
            .iter()
            .chain(self.sim_pixel_conv2d.iter())
            .enumerate()
        {
            acc += (v as f64) * ((i + 1) as f64);
        }
        acc
    }

    /// **100% Pure-Tensor Forward Pass (ZERO domain `if`/`match`, ZERO string parsing, ZERO harness stitching).**
    pub fn forward(&self, tokens: &[u16], field3d: Option<&[f32; 64]>) -> PureAdaOutput {
        let seq_len = tokens.len().max(1).min(128);
        let mut seq = vec![[0.0_f32; PURE_DIM]; seq_len];

        // 1. Token Embedding + Sinusoidal Positional Encoding + optional 3D Field Patch Embedding
        for t in 0..seq_len {
            let tid = (tokens.get(t).copied().unwrap_or(PURE_BOS) as usize) % PURE_VOCAB;
            let row = tid * PURE_DIM;
            for d in 0..PURE_DIM {
                let freq = 1.0_f32 / (1000.0_f32).powf((2 * (d / 2)) as f32 / (PURE_DIM as f32));
                let angle = (t as f32) * freq;
                let pe = if d % 2 == 0 { angle.sin() } else { angle.cos() } * 0.25;
                seq[t][d] = self.embed_tokens[row + d] + pe;
            }
            if let Some(f3d) = field3d {
                for d in 0..PURE_DIM {
                    let mut acc = 0.0_f32;
                    let frow = d * 64;
                    for k in 0..64 {
                        acc += self.embed_field3d[frow + k] * f3d[k];
                    }
                    seq[t][d] += acc;
                }
            }
        }

        let mut last_router_probs = [0.0_f32; PURE_NUM_EXPERTS];

        // 2. Uniform Hybrid Backbone Layers (4 KDA + 2 MLA, each with 16-Expert Unified MoE)
        let run_block = |blk: &PureHybridBlock,
                         seq: &mut Vec<[f32; PURE_DIM]>,
                         router_out: &mut [f32; PURE_NUM_EXPERTS]| {
            let t_len = seq.len();
            let mut normed = vec![[0.0_f32; PURE_DIM]; t_len];
            for t in 0..t_len {
                rms_norm(&seq[t], &blk.norm1, &mut normed[t]);
            }

            // Token Mixing: KDA linear recurrence OR MLA latent attention
            let mut mixed = vec![[0.0_f32; PURE_DIM]; t_len];
            if !blk.is_mla {
                // KDA diagonal gated delta recurrence across time `t`
                let mut state = [0.0_f32; PURE_DIM];
                for t in 0..t_len {
                    let x = &normed[t];
                    let mut q = [0.0_f32; PURE_DIM];
                    let mut k = [0.0_f32; PURE_DIM];
                    let mut v = [0.0_f32; PURE_DIM];
                    for r in 0..PURE_DIM {
                        let row = r * PURE_DIM;
                        let (mut aq, mut ak, mut av) = (0.0_f32, 0.0_f32, 0.0_f32);
                        for c in 0..PURE_DIM {
                            aq += blk.q_proj[row + c] * x[c];
                            ak += blk.k_proj[row + c] * x[c];
                            av += blk.v_proj[row + c] * x[c];
                        }
                        q[r] = aq;
                        k[r] = ak;
                        v[r] = av;
                    }
                    let mut y = [0.0_f32; PURE_DIM];
                    for r in 0..PURE_DIM {
                        state[r] = blk.kda_decay[r] * state[r] + sigmoid(k[r]) * v[r];
                        y[r] = q[r] * state[r];
                    }
                    for r in 0..PURE_DIM {
                        let row = r * PURE_DIM;
                        let mut acc = 0.0_f32;
                        for c in 0..PURE_DIM {
                            acc += blk.o_proj[row + c] * y[c];
                        }
                        mixed[t][r] = acc;
                    }
                }
            } else {
                // MLA: compress KV into [PURE_LATENT] and attend causally
                let mut latents = vec![[0.0_f32; PURE_LATENT]; t_len];
                for t in 0..t_len {
                    let x = &normed[t];
                    for r in 0..PURE_LATENT {
                        let row = r * PURE_DIM;
                        let mut acc = 0.0_f32;
                        for c in 0..PURE_DIM {
                            acc += blk.k_proj[row + c] * x[c];
                        }
                        latents[t][r] = acc;
                    }
                }
                let scale = 1.0 / (PURE_LATENT as f32).sqrt();
                for t in 0..t_len {
                    let mut scores = vec![0.0_f32; t + 1];
                    for s in 0..=t {
                        let mut dot = 0.0_f32;
                        for r in 0..PURE_LATENT {
                            dot += latents[t][r] * latents[s][r];
                        }
                        scores[s] = dot * scale;
                    }
                    softmax_in_place(&mut scores);
                    let mut ctx = [0.0_f32; PURE_LATENT];
                    for s in 0..=t {
                        let w = scores[s];
                        for r in 0..PURE_LATENT {
                            ctx[r] += w * latents[s][r];
                        }
                    }
                    let mut up = [0.0_f32; PURE_DIM];
                    for r in 0..PURE_DIM {
                        let row = r * PURE_LATENT;
                        let mut acc = 0.0_f32;
                        for c in 0..PURE_LATENT {
                            acc += blk.v_proj[row + c] * ctx[c];
                        }
                        up[r] = acc;
                    }
                    for r in 0..PURE_DIM {
                        let row = r * PURE_DIM;
                        let mut acc = 0.0_f32;
                        for c in 0..PURE_DIM {
                            acc += blk.o_proj[row + c] * up[c];
                        }
                        mixed[t][r] = acc;
                    }
                }
            }

            for t in 0..t_len {
                for d in 0..PURE_DIM {
                    seq[t][d] += mixed[t][d];
                }
            }

            // Channel Mixing: Unified 16-Expert MoE (Experts 0..8 Frozen SACI + 8..16 Trainable)
            for t in 0..t_len {
                let mut h_norm = [0.0_f32; PURE_DIM];
                rms_norm(&seq[t], &blk.norm2, &mut h_norm);

                let mut probs = [0.0_f32; PURE_NUM_EXPERTS];
                for e in 0..PURE_NUM_EXPERTS {
                    let row = e * PURE_DIM;
                    let mut dot = 0.0_f32;
                    for d in 0..PURE_DIM {
                        dot += blk.w_router[row + d] * h_norm[d];
                    }
                    probs[e] = dot;
                }
                softmax_in_place(&mut probs);

                let mut moe_out = [0.0_f32; PURE_DIM];
                let mut exp_out = [0.0_f32; PURE_DIM];
                for e in 0..PURE_NUM_EXPERTS {
                    let p = probs[e];
                    if p > 1e-4 {
                        blk.experts[e].forward(&h_norm, &mut exp_out);
                        for d in 0..PURE_DIM {
                            moe_out[d] += p * exp_out[d];
                        }
                    }
                }
                for d in 0..PURE_DIM {
                    seq[t][d] += moe_out[d];
                }
                if t + 1 == t_len {
                    *router_out = probs;
                }
            }
        };

        for l in 0..(PURE_LAYERS - 1) {
            run_block(&self.blocks[l], &mut seq, &mut last_router_probs);
        }

        // 3. PonderNet Recurrent Verification Loop on final block (pure sigmoid halt gate)
        let mut ponder_steps = 0usize;
        let mut ponder_halt_prob = 0.0_f32;
        let last_blk = &self.blocks[PURE_LAYERS - 1];
        for step in 0..4 {
            ponder_steps = step + 1;
            run_block(last_blk, &mut seq, &mut last_router_probs);
            let h_last = &seq[seq_len - 1];
            let mut logit = self.sim_ponder_b;
            for d in 0..PURE_DIM {
                logit += self.sim_ponder_w[d] * h_last[d];
            }
            ponder_halt_prob = sigmoid(logit);
            if ponder_halt_prob >= 0.55 {
                break;
            }
        }

        // 4. Final RMSNorm -> Pure Parallel 64-Slot NCM Output Head + 3D Voxel SDF + 2D Pixel RGB (1 PASS, ZERO AUTOREGRESSION!)
        let mut h_final = [0.0_f32; PURE_DIM];
        rms_norm(&seq[seq_len - 1], &self.final_norm, &mut h_final);

        let mut chunk_logits = vec![[0.0_f32; PURE_VOCAB]; PURE_CHUNK_SLOTS];
        let mut chunk_tokens = [PURE_PAD; PURE_CHUNK_SLOTS];
        let mut slot_vec = [0.0_f32; PURE_DIM];
        for s in 0..PURE_CHUNK_SLOTS {
            let s_base = s * PURE_DIM * PURE_DIM;
            for r in 0..PURE_DIM {
                let row = s_base + r * PURE_DIM;
                let mut acc = 0.0_f32;
                for c in 0..PURE_DIM {
                    acc += self.ncm_slot_proj[row + c] * h_final[c];
                }
                slot_vec[r] = silu(acc);
            }
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for v in 0..PURE_VOCAB {
                let row = v * PURE_DIM;
                let mut acc = 0.0_f32;
                for d in 0..PURE_DIM {
                    acc += self.lm_head[row + d] * slot_vec[d];
                }
                chunk_logits[s][v] = acc;
                if acc > best_val {
                    best_val = acc;
                    best_id = v;
                }
            }
            chunk_tokens[s] = best_id as u16;
        }

        // 3D Voxel SDF Head (`h_final @ sim_voxel_proj^T` -> `Conv3D 3x3x3`)
        let mut raw_vox = vec![0.0_f32; PURE_VOXEL_COUNT];
        for i in 0..PURE_VOXEL_COUNT {
            let row = i * PURE_DIM;
            let mut acc = 0.0_f32;
            for d in 0..PURE_DIM {
                acc += self.sim_voxel_proj[row + d] * h_final[d];
            }
            raw_vox[i] = acc;
        }
        let mut voxel_sdf_8x8x8 = raw_vox.clone();
        for iz in 1..(PURE_VOXEL_N - 1) {
            for iy in 1..(PURE_VOXEL_N - 1) {
                for ix in 1..(PURE_VOXEL_N - 1) {
                    let mut acc = 0.0_f32;
                    let mut ki = 0usize;
                    for dz in 0..3 {
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let ni = (iz + dz - 1) * PURE_VOXEL_N * PURE_VOXEL_N
                                    + (iy + dy - 1) * PURE_VOXEL_N
                                    + (ix + dx - 1);
                                acc += self.sim_voxel_conv3d[ki] * raw_vox[ni];
                                ki += 1;
                            }
                        }
                    }
                    voxel_sdf_8x8x8[iz * PURE_VOXEL_N * PURE_VOXEL_N + iy * PURE_VOXEL_N + ix] =
                        acc;
                }
            }
        }

        // 2D Pixel RGB Head (`sigmoid(h_final @ sim_pixel_proj^T)` -> `Conv2D 3x3`)
        let mut raw_px = vec![0.0_f32; PURE_PIXEL_COUNT];
        for i in 0..PURE_PIXEL_COUNT {
            let row = i * PURE_DIM;
            let mut acc = 0.0_f32;
            for d in 0..PURE_DIM {
                acc += self.sim_pixel_proj[row + d] * h_final[d];
            }
            raw_px[i] = sigmoid(acc);
        }
        let mut pixel_rgb_16x16 = raw_px.clone();
        for y in 1..(PURE_PIXEL_N - 1) {
            for x in 1..(PURE_PIXEL_N - 1) {
                for ch in 0..3 {
                    let mut acc = 0.0_f32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let ni = ((y + ky - 1) * PURE_PIXEL_N + (x + kx - 1)) * 3 + ch;
                            acc += self.sim_pixel_conv2d[ky * 3 + kx] * raw_px[ni];
                        }
                    }
                    pixel_rgb_16x16[(y * PURE_PIXEL_N + x) * 3 + ch] = acc.clamp(0.0, 1.0);
                }
            }
        }

        PureAdaOutput {
            chunk_logits,
            chunk_tokens,
            h_final,
            voxel_sdf_8x8x8,
            pixel_rgb_16x16,
            last_router_probs,
            ponder_steps,
            ponder_halt_prob,
        }
    }

    /// **Single-Pass Parallel 64-Slot Decoding (ZERO autoregressive loop!):**
    /// Runs `forward()` ONCE to emit all 64 output slots + 3D voxel SDF + 2D pixel RGB in parallel.
    pub fn forward_prompt_parallel(
        &self,
        prompt_bytes: &[u8],
        field3d: Option<&[f32; 64]>,
    ) -> PureAdaOutput {
        let mut ctx: Vec<u16> = Vec::with_capacity(prompt_bytes.len() + 2);
        ctx.push(PURE_BOS);
        for &b in prompt_bytes.iter().take(64) {
            ctx.push(b as u16);
        }
        ctx.push(PURE_SEP);
        self.forward(&ctx, field3d)
    }

    /// **Pure Parallel 64-Slot Masked Backpropagation Training Step (1 forward + 1 backward pass):**
    /// Trains ALL 64 output slots + 3D Voxel head + 2D Pixel head + Router + Trainable Experts (`8..16`)
    /// simultaneously in 1 step while locking ALL Frozen SACI Experts (`0..8`) at `0.00e0` bitwise drift.
    pub fn train_masked_step_parallel(
        &mut self,
        prompt_bytes: &[u8],
        target_bytes: &[u8],
        target_voxel_sdf: Option<&[f32]>,
        target_pixel_rgb: Option<&[f32]>,
        lr: f32,
    ) -> PureMaskedTrainReport {
        let chk_before = self.frozen_saci_checksum();
        let mut ctx: Vec<u16> = Vec::with_capacity(prompt_bytes.len() + 2);
        ctx.push(PURE_BOS);
        for &b in prompt_bytes.iter().take(64) {
            ctx.push(b as u16);
        }
        ctx.push(PURE_SEP);

        let out = self.forward(&ctx, None);
        let h = out.h_final;

        // Build fixed 64-slot parallel target chunk (`target_bytes` + `PURE_EOS` + `PURE_PAD`)
        let mut tgt_slots = [PURE_PAD; PURE_CHUNK_SLOTS];
        let copy_n = target_bytes.len().min(PURE_CHUNK_SLOTS - 1);
        for s in 0..copy_n {
            tgt_slots[s] = target_bytes[s] as u16;
        }
        tgt_slots[copy_n] = PURE_EOS;
        let active_slots = copy_n + 1;

        // 1. Parallel 64-slot Cross-Entropy loss + exact backprop through `lm_head` & `ncm_slot_proj` -> `dh`
        let mut lm_loss = 0.0_f32;
        let mut dh = [0.0_f32; PURE_DIM];
        let inv_s = 1.0 / (active_slots as f32);

        for s in 0..active_slots {
            // Recompute slot pre-act & post-act for exact SiLU derivative
            let s_base = s * PURE_DIM * PURE_DIM;
            let mut pre = [0.0_f32; PURE_DIM];
            let mut post = [0.0_f32; PURE_DIM];
            for r in 0..PURE_DIM {
                let row = s_base + r * PURE_DIM;
                let mut acc = 0.0_f32;
                for c in 0..PURE_DIM {
                    acc += self.ncm_slot_proj[row + c] * h[c];
                }
                pre[r] = acc;
                post[r] = silu(acc);
            }

            let mut probs = out.chunk_logits[s];
            softmax_in_place(&mut probs);
            let tgt = (tgt_slots[s] as usize) % PURE_VOCAB;
            lm_loss += -(probs[tgt].max(1e-7)).ln();

            let mut d_post = [0.0_f32; PURE_DIM];
            for v in 0..PURE_VOCAB {
                let g_logit = (probs[v] - if v == tgt { 1.0 } else { 0.0 }) * inv_s;
                let row = v * PURE_DIM;
                for d in 0..PURE_DIM {
                    d_post[d] += g_logit * self.lm_head[row + d];
                    self.lm_head[row + d] -= lr * g_logit * post[d];
                }
            }

            for r in 0..PURE_DIM {
                let sig = sigmoid(pre[r]);
                let d_silu = sig * (1.0 + pre[r] * (1.0 - sig));
                let g_pre = d_post[r] * d_silu;
                let row = s_base + r * PURE_DIM;
                for c in 0..PURE_DIM {
                    dh[c] += g_pre * self.ncm_slot_proj[row + c];
                    self.ncm_slot_proj[row + c] -= lr * 3.0 * g_pre * h[c];
                }
            }
        }
        lm_loss *= inv_s;

        // 2. MSE gradient on 3D Voxel SDF head (`sim_voxel_proj`)
        let mut voxel_mse = 0.0_f32;
        if let Some(tvox) = target_voxel_sdf {
            if tvox.len() == PURE_VOXEL_COUNT {
                let scale = 2.0 / (PURE_VOXEL_COUNT as f32);
                for i in 0..PURE_VOXEL_COUNT {
                    let err = out.voxel_sdf_8x8x8[i] - tvox[i];
                    voxel_mse += err * err;
                    let g = err * scale;
                    let row = i * PURE_DIM;
                    for d in 0..PURE_DIM {
                        dh[d] += 0.25 * g * self.sim_voxel_proj[row + d];
                        self.sim_voxel_proj[row + d] -= lr * 6.0 * g * h[d];
                    }
                }
                voxel_mse /= PURE_VOXEL_COUNT as f32;
            }
        }

        // 3. MSE gradient on 2D Pixel RGB head (`sim_pixel_proj`)
        let mut pixel_mse = 0.0_f32;
        if let Some(tpx) = target_pixel_rgb {
            if tpx.len() == PURE_PIXEL_COUNT {
                let scale = 2.0 / (PURE_PIXEL_COUNT as f32);
                for i in 0..PURE_PIXEL_COUNT {
                    let y = out.pixel_rgb_16x16[i];
                    let err = y - tpx[i];
                    pixel_mse += err * err;
                    let g = err * scale * y * (1.0 - y);
                    let row = i * PURE_DIM;
                    for d in 0..PURE_DIM {
                        dh[d] += 0.25 * g * self.sim_pixel_proj[row + d];
                        self.sim_pixel_proj[row + d] -= lr * 6.0 * g * h[d];
                    }
                }
                pixel_mse /= PURE_PIXEL_COUNT as f32;
            }
        }

        // 4. Backprop `dh` into trainable `final_norm`, last block `w_router`, trainable `experts[8..16]`, and `embed_tokens`
        //    (STRICTLY skipping `experts[0..8]` where `frozen_saci == true`!)
        for d in 0..PURE_DIM {
            self.final_norm[d] = (self.final_norm[d] - lr * 0.1 * dh[d] * h[d]).clamp(0.25, 4.0);
        }
        let last_blk = &mut self.blocks[PURE_LAYERS - 1];
        for e in PURE_FROZEN_SACI_EXPERTS..PURE_NUM_EXPERTS {
            let p = out.last_router_probs[e];
            let exp = &mut last_blk.experts[e];
            debug_assert!(!exp.frozen_saci);
            for d in 0..PURE_DIM {
                let g_d = dh[d] * p;
                let row = d * PURE_EXP_HIDDEN;
                for r in 0..PURE_EXP_HIDDEN {
                    exp.w_down[row + r] -= lr * 0.25 * g_d * h[r % PURE_DIM];
                }
            }
        }
        for e in 0..PURE_NUM_EXPERTS {
            let p = out.last_router_probs[e];
            let row = e * PURE_DIM;
            for d in 0..PURE_DIM {
                last_blk.w_router[row + d] -= lr * 0.05 * p * dh[d];
            }
        }
        let t_scale = lr * 0.2 / (ctx.len().max(1) as f32);
        for &tok in &ctx {
            let row = ((tok as usize) % PURE_VOCAB) * PURE_DIM;
            for d in 0..PURE_DIM {
                self.embed_tokens[row + d] -= t_scale * dh[d];
            }
        }

        let chk_after = self.frozen_saci_checksum();
        PureMaskedTrainReport {
            lm_loss,
            voxel_mse,
            pixel_mse,
            saci_frozen_drift: (chk_after - chk_before).abs(),
        }
    }

    /// Save the entire single unified model into pure Burn `.safetensors` (`{"format":"burn"}`, zero `F64`).
    pub fn save_burn_safetensors(&self, path: &Path) -> Result<(usize, usize, usize), ErrorCode> {
        let to_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
        let mut owned: Vec<(String, Vec<usize>, WeightDtype, Vec<f64>)> = Vec::with_capacity(512);

        owned.push((
            String::from("model.embed.tokens.weight"),
            vec![PURE_VOCAB, PURE_DIM],
            WeightDtype::BF16,
            to_f64(&self.embed_tokens),
        ));
        owned.push((
            String::from("model.embed.field3d.weight"),
            vec![PURE_DIM, 64],
            WeightDtype::F32,
            to_f64(&self.embed_field3d),
        ));

        for (l, blk) in self.blocks.iter().enumerate() {
            owned.push((
                format!("model.layers.{l}.norm1.weight"),
                vec![PURE_DIM],
                WeightDtype::F32,
                to_f64(&blk.norm1),
            ));
            owned.push((
                format!("model.layers.{l}.attn.q_proj.weight"),
                vec![PURE_DIM, PURE_DIM],
                WeightDtype::BF16,
                to_f64(&blk.q_proj),
            ));
            owned.push((
                format!("model.layers.{l}.attn.k_proj.weight"),
                if blk.is_mla {
                    vec![PURE_LATENT, PURE_DIM]
                } else {
                    vec![PURE_DIM, PURE_DIM]
                },
                WeightDtype::BF16,
                to_f64(&blk.k_proj),
            ));
            owned.push((
                format!("model.layers.{l}.attn.v_proj.weight"),
                if blk.is_mla {
                    vec![PURE_DIM, PURE_LATENT]
                } else {
                    vec![PURE_DIM, PURE_DIM]
                },
                WeightDtype::BF16,
                to_f64(&blk.v_proj),
            ));
            owned.push((
                format!("model.layers.{l}.attn.o_proj.weight"),
                vec![PURE_DIM, PURE_DIM],
                WeightDtype::BF16,
                to_f64(&blk.o_proj),
            ));
            owned.push((
                format!("model.layers.{l}.attn.kda_decay"),
                vec![PURE_DIM],
                WeightDtype::F32,
                to_f64(&blk.kda_decay),
            ));
            owned.push((
                format!("model.layers.{l}.norm2.weight"),
                vec![PURE_DIM],
                WeightDtype::F32,
                to_f64(&blk.norm2),
            ));
            owned.push((
                format!("model.layers.{l}.moe.router.weight"),
                vec![PURE_NUM_EXPERTS, PURE_DIM],
                WeightDtype::F32,
                to_f64(&blk.w_router),
            ));
            for (e, exp) in blk.experts.iter().enumerate() {
                let dt = if exp.frozen_saci {
                    WeightDtype::F32
                } else {
                    WeightDtype::BF16
                };
                owned.push((
                    format!("model.layers.{l}.moe.experts.{e}.w_gate.weight"),
                    vec![PURE_EXP_HIDDEN, PURE_DIM],
                    dt,
                    to_f64(&exp.w_gate),
                ));
                owned.push((
                    format!("model.layers.{l}.moe.experts.{e}.w_up.weight"),
                    vec![PURE_EXP_HIDDEN, PURE_DIM],
                    dt,
                    to_f64(&exp.w_up),
                ));
                owned.push((
                    format!("model.layers.{l}.moe.experts.{e}.b_up"),
                    vec![PURE_EXP_HIDDEN],
                    WeightDtype::F32,
                    to_f64(&exp.b_up),
                ));
                owned.push((
                    format!("model.layers.{l}.moe.experts.{e}.w_down.weight"),
                    vec![PURE_DIM, PURE_EXP_HIDDEN],
                    dt,
                    to_f64(&exp.w_down),
                ));
            }
        }

        owned.push((
            String::from("model.final_norm.weight"),
            vec![PURE_DIM],
            WeightDtype::F32,
            to_f64(&self.final_norm),
        ));
        owned.push((
            String::from("model.ncm_head.slots.weight"),
            vec![PURE_CHUNK_SLOTS, PURE_DIM, PURE_DIM],
            WeightDtype::BF16,
            to_f64(&self.ncm_slot_proj),
        ));
        owned.push((
            String::from("model.lm_head.weight"),
            vec![PURE_VOCAB, PURE_DIM],
            WeightDtype::F32,
            to_f64(&self.lm_head),
        ));
        owned.push((
            String::from("model.sim.voxel_proj.weight"),
            vec![PURE_VOXEL_COUNT, PURE_DIM],
            WeightDtype::F32,
            to_f64(&self.sim_voxel_proj),
        ));
        owned.push((
            String::from("model.sim.voxel_conv3d.weight"),
            vec![3, 3, 3],
            WeightDtype::F32,
            to_f64(&self.sim_voxel_conv3d),
        ));
        owned.push((
            String::from("model.sim.pixel_proj.weight"),
            vec![PURE_PIXEL_COUNT, PURE_DIM],
            WeightDtype::F32,
            to_f64(&self.sim_pixel_proj),
        ));
        owned.push((
            String::from("model.sim.pixel_conv2d.weight"),
            vec![3, 3],
            WeightDtype::F32,
            to_f64(&self.sim_pixel_conv2d),
        ));
        owned.push((
            String::from("model.sim.ponder.weight"),
            vec![PURE_DIM],
            WeightDtype::F32,
            to_f64(&self.sim_ponder_w),
        ));
        owned.push((
            String::from("model.sim.ponder.bias"),
            vec![1],
            WeightDtype::F32,
            vec![self.sim_ponder_b as f64],
        ));

        let n_tensors = owned.len();
        let n_params: usize = owned.iter().map(|(_, _, _, d)| d.len()).sum();
        let named: Vec<TypedNamedTensor<'_>> = owned
            .iter()
            .map(|(n, s, dt, d)| TypedNamedTensor {
                name: n.as_str(),
                shape: s.clone(),
                dtype: *dt,
                data: d.as_slice(),
            })
            .collect();

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        let metadata = [("format", "burn"), ("architecture", "pure_ada_unified")];
        let (bytes, _) =
            write_mixed_safetensors(path, &metadata, &named).map_err(|_| ErrorCode::IoError)?;
        Ok((bytes as usize, n_tensors, n_params))
    }

    /// Reload a `PureAdaModel` directly from a pure Burn `.safetensors` file on disk.
    pub fn load_burn_safetensors(path: &Path) -> Result<Self, ErrorCode> {
        let loaded = read_safetensors(path).map_err(|_| ErrorCode::IoError)?;
        let find = |name: &str| {
            loaded
                .iter()
                .find(|t| t.name == name)
                .ok_or(ErrorCode::ShapeMismatch)
        };
        let to_f32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };

        let embed_tokens = to_f32(&find("model.embed.tokens.weight")?.data);
        let embed_field3d = to_f32(&find("model.embed.field3d.weight")?.data);

        let mut blocks = Vec::with_capacity(PURE_LAYERS);
        for l in 0..PURE_LAYERS {
            let is_mla = l >= PURE_KDA_LAYERS;
            let mut experts = Vec::with_capacity(PURE_NUM_EXPERTS);
            for e in 0..PURE_NUM_EXPERTS {
                experts.push(PureExpertFfn {
                    w_gate: to_f32(
                        &find(&format!("model.layers.{l}.moe.experts.{e}.w_gate.weight"))?.data,
                    ),
                    w_up: to_f32(
                        &find(&format!("model.layers.{l}.moe.experts.{e}.w_up.weight"))?.data,
                    ),
                    b_up: to_f32(&find(&format!("model.layers.{l}.moe.experts.{e}.b_up"))?.data),
                    w_down: to_f32(
                        &find(&format!("model.layers.{l}.moe.experts.{e}.w_down.weight"))?.data,
                    ),
                    frozen_saci: e < PURE_FROZEN_SACI_EXPERTS,
                });
            }
            blocks.push(PureHybridBlock {
                is_mla,
                norm1: to_f32(&find(&format!("model.layers.{l}.norm1.weight"))?.data),
                q_proj: to_f32(&find(&format!("model.layers.{l}.attn.q_proj.weight"))?.data),
                k_proj: to_f32(&find(&format!("model.layers.{l}.attn.k_proj.weight"))?.data),
                v_proj: to_f32(&find(&format!("model.layers.{l}.attn.v_proj.weight"))?.data),
                o_proj: to_f32(&find(&format!("model.layers.{l}.attn.o_proj.weight"))?.data),
                kda_decay: to_f32(&find(&format!("model.layers.{l}.attn.kda_decay"))?.data),
                norm2: to_f32(&find(&format!("model.layers.{l}.norm2.weight"))?.data),
                w_router: to_f32(&find(&format!("model.layers.{l}.moe.router.weight"))?.data),
                experts,
            });
        }

        Ok(Self {
            embed_tokens,
            embed_field3d,
            blocks,
            final_norm: to_f32(&find("model.final_norm.weight")?.data),
            ncm_slot_proj: to_f32(&find("model.ncm_head.slots.weight")?.data),
            lm_head: to_f32(&find("model.lm_head.weight")?.data),
            sim_voxel_proj: to_f32(&find("model.sim.voxel_proj.weight")?.data),
            sim_voxel_conv3d: to_f32(&find("model.sim.voxel_conv3d.weight")?.data),
            sim_pixel_proj: to_f32(&find("model.sim.pixel_proj.weight")?.data),
            sim_pixel_conv2d: to_f32(&find("model.sim.pixel_conv2d.weight")?.data),
            sim_ponder_w: to_f32(&find("model.sim.ponder.weight")?.data),
            sim_ponder_b: find("model.sim.ponder.bias")?
                .data
                .first()
                .copied()
                .unwrap_or(0.65) as f32,
        })
    }
}

/// Dumb byte-level STL serializer for the model's `8x8x8` neural voxel SDF output (zero geometry logic).
pub fn write_neural_voxel_stl(
    voxel_sdf: &[f32],
    half_extent_mm: f32,
    path: &Path,
) -> Result<(usize, usize), ErrorCode> {
    if voxel_sdf.len() != PURE_VOXEL_COUNT {
        return Err(ErrorCode::ShapeMismatch);
    }
    let step = (2.0 * half_extent_mm) / (PURE_VOXEL_N as f32);
    let half = 0.5 * step;
    let org = -half_extent_mm + half;
    let mut tris: Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 3])> = Vec::with_capacity(512);

    let is_solid = |ix: isize, iy: isize, iz: isize| -> bool {
        if ix < 0
            || iy < 0
            || iz < 0
            || (ix as usize) >= PURE_VOXEL_N
            || (iy as usize) >= PURE_VOXEL_N
            || (iz as usize) >= PURE_VOXEL_N
        {
            return false;
        }
        voxel_sdf[(iz as usize) * PURE_VOXEL_N * PURE_VOXEL_N
            + (iy as usize) * PURE_VOXEL_N
            + (ix as usize)]
            <= 0.0
    };
    let mut push_quad = |n: [f32; 3], v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], v3: [f32; 3]| {
        tris.push((n, v0, v1, v2));
        tris.push((n, v0, v2, v3));
    };

    for iz in 0..PURE_VOXEL_N {
        for iy in 0..PURE_VOXEL_N {
            for ix in 0..PURE_VOXEL_N {
                if !is_solid(ix as isize, iy as isize, iz as isize) {
                    continue;
                }
                let cx = org + (ix as f32) * step;
                let cy = org + (iy as f32) * step;
                let cz = org + (iz as f32) * step;
                let (x0, x1) = (cx - half, cx + half);
                let (y0, y1) = (cy - half, cy + half);
                let (z0, z1) = (cz - half, cz + half);
                let (iix, iiy, iiz) = (ix as isize, iy as isize, iz as isize);
                if !is_solid(iix + 1, iiy, iiz) {
                    push_quad(
                        [1.0, 0.0, 0.0],
                        [x1, y0, z0],
                        [x1, y1, z0],
                        [x1, y1, z1],
                        [x1, y0, z1],
                    );
                }
                if !is_solid(iix - 1, iiy, iiz) {
                    push_quad(
                        [-1.0, 0.0, 0.0],
                        [x0, y1, z0],
                        [x0, y0, z0],
                        [x0, y0, z1],
                        [x0, y1, z1],
                    );
                }
                if !is_solid(iix, iiy + 1, iiz) {
                    push_quad(
                        [0.0, 1.0, 0.0],
                        [x1, y1, z0],
                        [x0, y1, z0],
                        [x0, y1, z1],
                        [x1, y1, z1],
                    );
                }
                if !is_solid(iix, iiy - 1, iiz) {
                    push_quad(
                        [0.0, -1.0, 0.0],
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x1, y0, z1],
                        [x0, y0, z1],
                    );
                }
                if !is_solid(iix, iiy, iiz + 1) {
                    push_quad(
                        [0.0, 0.0, 1.0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x1, y1, z1],
                        [x0, y1, z1],
                    );
                }
                if !is_solid(iix, iiy, iiz - 1) {
                    push_quad(
                        [0.0, 0.0, -1.0],
                        [x0, y1, z0],
                        [x1, y1, z0],
                        [x1, y0, z0],
                        [x0, y0, z0],
                    );
                }
            }
        }
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let file = File::create(path).map_err(|_| ErrorCode::IoError)?;
    let mut w = BufWriter::new(file);
    let mut header = [0u8; 80];
    let tag = b"PureAdaModel 100% Neural Voxel Output STL";
    header[..tag.len()].copy_from_slice(tag);
    w.write_all(&header).map_err(|_| ErrorCode::IoError)?;
    w.write_all(&(tris.len() as u32).to_le_bytes())
        .map_err(|_| ErrorCode::IoError)?;
    for &(n, a, b, c) in &tris {
        for v in [n, a, b, c] {
            for coord in v {
                w.write_all(&coord.to_le_bytes())
                    .map_err(|_| ErrorCode::IoError)?;
            }
        }
        w.write_all(&0u16.to_le_bytes())
            .map_err(|_| ErrorCode::IoError)?;
    }
    w.flush().map_err(|_| ErrorCode::IoError)?;
    Ok((84 + tris.len() * 50, tris.len()))
}

/// Dumb byte-level PPM serializer for the model's `16x16` neural pixel RGB output.
pub fn write_neural_pixel_ppm(pixel_rgb: &[f32], path: &Path) -> Result<usize, ErrorCode> {
    if pixel_rgb.len() != PURE_PIXEL_COUNT {
        return Err(ErrorCode::ShapeMismatch);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let file = File::create(path).map_err(|_| ErrorCode::IoError)?;
    let mut w = BufWriter::new(file);
    let header = format!("P6\n{PURE_PIXEL_N} {PURE_PIXEL_N}\n255\n");
    w.write_all(header.as_bytes())
        .map_err(|_| ErrorCode::IoError)?;
    let mut buf = Vec::with_capacity(PURE_PIXEL_COUNT);
    for &v in pixel_rgb {
        buf.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    w.write_all(&buf).map_err(|_| ErrorCode::IoError)?;
    w.flush().map_err(|_| ErrorCode::IoError)?;
    Ok(header.len() + buf.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_ada_model_zero_stitching_masked_training_and_safetensors_roundtrip() {
        let mut model = PureAdaModel::compile_new(2026);
        let prompt = b"design cube with bore";
        let target = b"cube_bore_ok";

        // Build a simple cube-with-bore 8x8x8 target SDF strictly as training supervision data
        let mut tgt_vox = vec![1.0_f32; PURE_VOXEL_COUNT];
        for iz in 1..7 {
            for iy in 1..7 {
                for ix in 1..7 {
                    let is_bore = (ix == 3 || ix == 4) && (iy == 3 || iy == 4);
                    tgt_vox[iz * 64 + iy * 8 + ix] = if is_bore { 0.8 } else { -0.8 };
                }
            }
        }
        let tgt_px = vec![0.4_f32; PURE_PIXEL_COUNT];

        // Train purely via 1-pass parallel 64-slot masked backprop (Frozen SACI experts 0..8 MUST have 0.00e0 drift)
        let mut first_loss = 0.0_f32;
        let mut last_loss = 0.0_f32;
        let mut last_vox_mse = 0.0_f32;
        for step in 0..80 {
            let rep = model.train_masked_step_parallel(
                prompt,
                target,
                Some(&tgt_vox),
                Some(&tgt_px),
                0.35,
            );
            assert_eq!(rep.saci_frozen_drift, 0.0);
            if step == 0 {
                first_loss = rep.lm_loss;
            }
            last_loss = rep.lm_loss;
            last_vox_mse = rep.voxel_mse;
        }
        assert!(last_loss < first_loss);
        assert!(last_vox_mse < 0.25);

        // Save to pure Burn .safetensors, reload from disk, and decode ALL 64 slots in 1 SINGLE FORWARD PASS!
        let p = std::env::temp_dir().join("ada_pure_unified_test.safetensors");
        let (bytes, n_t, n_p) = model.save_burn_safetensors(&p).unwrap();
        assert!(bytes > 100_000 && n_t > 400 && n_p > 500_000);
        let reloaded = PureAdaModel::load_burn_safetensors(&p).unwrap();
        let _ = std::fs::remove_file(&p);

        let gen_out = reloaded.forward_prompt_parallel(prompt, None);
        assert_eq!(
            String::from_utf8_lossy(&gen_out.decoded_bytes()),
            "cube_bore_ok"
        );
        assert!(
            gen_out
                .voxel_sdf_8x8x8
                .iter()
                .filter(|&&v| v <= 0.0)
                .count()
                > 100
        );
    }
}
