//! Multi-Turn Conversational State, Multi-Chunk NCM Continuation & Pure-SafeTensors Loader
//! (`conversation.rs`).
//!
//! Fixes Gap 2 (single 64-token chunk & stateless turns):
//! - `LoadedAdaModel::from_safetensors` loads all 58 pure-weight tensors (including `VqDecoderWeights`)
//!   directly from `.safetensors` with zero `.bpk` dependency at inference time.
//! - `ConversationSession` maintains a fixed-capacity KDA recurrent memory matrix `M in R^{dim x dim}`
//!   (bounded memory, zero unbounded KV-cache growth) across turns and supports multi-chunk
//!   continuation (`1..=max_chunks` × 64 tokens) with `[SEP]` / plateau halting.

use std::fs;
use std::path::{Path, PathBuf};

use crate::attention::{encode_prompt_matrix, encode_token_slots};
use crate::checkpoint::Checkpoint;
use crate::knowledge_base::{
    forward_pure_weights_with_saci, normalize_vec, PureWeightAdaModel, KB_DIM,
};
use crate::ncm_generator::{
    ParallelNcmGenerator, TransformerDecBlock, VqDecoderWeights, NCM_CHUNK_LEN,
};
use crate::operand_grounding::NumericVocabTable;
use crate::saci_compiler::{
    compile_saci_program, load_saci_binary, parse_saci_source, SaciCircuitDescriptor,
    SacisubstrateAllocator,
};
use crate::safetensors_writer::{read_safetensors, LoadedTensor};
use crate::sequence_encoder::{SequenceEncoder, SequenceScratch};
use crate::sys::ErrorCode;

const TOK_SEP: u16 = 102;
const TOK_PAD: u16 = 0;

fn find_t<'a>(tensors: &'a [LoadedTensor], name: &str) -> Result<&'a LoadedTensor, ErrorCode> {
    tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or(ErrorCode::InvalidInput)
}

fn load_vq_block(tensors: &[LoadedTensor], prefix: &str) -> Result<TransformerDecBlock, ErrorCode> {
    let ff1 = find_t(tensors, &format!("{prefix}.ff1.weight"))?;
    if ff1.shape.len() != 2 {
        return Err(ErrorCode::ShapeMismatch);
    }
    Ok(TransformerDecBlock {
        ln1_gamma: find_t(tensors, &format!("{prefix}.ln1.gamma"))?
            .data
            .clone(),
        ln1_beta: find_t(tensors, &format!("{prefix}.ln1.beta"))?.data.clone(),
        wq: find_t(tensors, &format!("{prefix}.wq.weight"))?
            .data
            .clone(),
        bq: find_t(tensors, &format!("{prefix}.wq.bias"))?.data.clone(),
        wk: find_t(tensors, &format!("{prefix}.wk.weight"))?
            .data
            .clone(),
        bk: find_t(tensors, &format!("{prefix}.wk.bias"))?.data.clone(),
        wv: find_t(tensors, &format!("{prefix}.wv.weight"))?
            .data
            .clone(),
        bv: find_t(tensors, &format!("{prefix}.wv.bias"))?.data.clone(),
        wo: find_t(tensors, &format!("{prefix}.wo.weight"))?
            .data
            .clone(),
        bo: find_t(tensors, &format!("{prefix}.wo.bias"))?.data.clone(),
        ln2_gamma: find_t(tensors, &format!("{prefix}.ln2.gamma"))?
            .data
            .clone(),
        ln2_beta: find_t(tensors, &format!("{prefix}.ln2.beta"))?.data.clone(),
        ff1_w: ff1.data.clone(),
        ff1_b: find_t(tensors, &format!("{prefix}.ff1.bias"))?.data.clone(),
        ff2_w: find_t(tensors, &format!("{prefix}.ff2.weight"))?
            .data
            .clone(),
        ff2_b: find_t(tensors, &format!("{prefix}.ff2.bias"))?.data.clone(),
        ff_dim: ff1.shape[1],
    })
}

/// Complete self-contained ADA model loaded directly from `.safetensors` (+ optional `.saci` descriptors).
pub struct LoadedAdaModel {
    pub dim: usize,
    pub vocab_size: usize,
    pub pure_model: PureWeightAdaModel,
    pub ncm_gen: ParallelNcmGenerator,
    pub vq_dec: VqDecoderWeights,
    pub vq_tok_emb: Vec<f64>,
    pub saci_ckpt: Checkpoint,
    pub saci_descs: Vec<SaciCircuitDescriptor>,
}

impl LoadedAdaModel {
    pub fn from_safetensors(model_path: &Path, saci_dir: Option<&Path>) -> Result<Self, ErrorCode> {
        let tensors = read_safetensors(model_path).map_err(|_| ErrorCode::InvalidInput)?;
        let emb_t = find_t(&tensors, "vq.tok_emb.weight")?;
        if emb_t.shape.len() != 2 || emb_t.shape[1] != KB_DIM {
            return Err(ErrorCode::ShapeMismatch);
        }
        let vocab_size = emb_t.shape[0];
        let dim = emb_t.shape[1];
        let vq_tok_emb = emb_t.data.clone();

        let vh = &find_t(&tensors, "bridges.verify_halt")?.data;
        if vh.len() < 4 {
            return Err(ErrorCode::ShapeMismatch);
        }
        let pure_model = PureWeightAdaModel {
            w_gate: find_t(&tensors, "bridges.w_gate")?.data.clone(),
            b_gate: find_t(&tensors, "bridges.b_gate")?.data.clone(),
            w_knowledge: find_t(&tensors, "brain.kda.w_knowledge")?.data.clone(),
            w_reasoning: find_t(&tensors, "brain.mla.w_reasoning")?.data.clone(),
            w_saci_route: find_t(&tensors, "saci.router.w_route")?.data.clone(),
            w_saci_facts: find_t(&tensors, "saci.facts.w_facts")?.data.clone(),
            w_simulation: find_t(&tensors, "workspace.w_simulation")?.data.clone(),
            p_knowledge: find_t(&tensors, "brain.kda.p_knowledge")?.data.clone(),
            p_reasoning: find_t(&tensors, "brain.mla.p_reasoning")?.data.clone(),
            p_saci: find_t(&tensors, "saci.router.p_saci")?.data.clone(),
            p_simulation: find_t(&tensors, "workspace.p_simulation")?.data.clone(),
            verify_bias: vh[0],
            w_halt: [vh[1], vh[2]],
            b_halt: vh[3],
        };

        let dm_t = find_t(&tensors, "ncm.domain_means")?;
        let num_domains = dm_t.shape[0];
        let w_exp_t = find_t(&tensors, "ncm.w_expand")?;
        let exp_dim = w_exp_t.shape[1];
        let w_coarse_t = find_t(&tensors, "ncm.w_coarse")?;
        let coarse_len = w_coarse_t.shape[1] / dim;

        let ncm_gen = ParallelNcmGenerator {
            ws_dim: 12 * dim,
            exp_dim,
            dom_exp_dim: 0,
            num_domains,
            seq_len: NCM_CHUNK_LEN,
            dim,
            w_expand: w_exp_t.data.clone(),
            w_dom_expand: Vec::new(),
            domain_means: dm_t.data.clone(),
            w_ncm_chunk: find_t(&tensors, "ncm.w_ncm_chunk")?.data.clone(),
            w_ncm_domain: Vec::new(),
            a_copy: Vec::new(),
            w_val: Vec::new(),
            coarse_len,
            w_coarse: w_coarse_t.data.clone(),
            w_q_attn: find_t(&tensors, "ncm.w_q_attn")?.data.clone(),
            w_k_heads: find_t(&tensors, "ncm.w_k_heads")?.data.clone(),
            rel_pos_bias: find_t(&tensors, "ncm.rel_pos_bias")?.data.clone(),
            w_v_attn: find_t(&tensors, "ncm.w_v_attn")?.data.clone(),
        };

        let vq_dec = VqDecoderWeights {
            dim,
            vocab_size,
            dec1: load_vq_block(&tensors, "vq.dec1")?,
            dec2: load_vq_block(&tensors, "vq.dec2")?,
            ln_dec_gamma: find_t(&tensors, "vq.ln_dec.gamma")?.data.clone(),
            ln_dec_beta: find_t(&tensors, "vq.ln_dec.beta")?.data.clone(),
            tok_emb: vq_tok_emb.clone(),
            head_bias: find_t(&tensors, "vq.head_bias")?.data.clone(),
        };

        let mut saci_ckpt = Checkpoint::new(Vec::new(), Vec::new())?;
        let mut saci_descs = Vec::new();
        if let Some(dir) = saci_dir {
            if let Ok(rd) = fs::read_dir(dir) {
                let mut files: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok().map(|de| de.path()))
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("saci"))
                    .collect();
                files.sort();
                let mut allocator = SacisubstrateAllocator::new(2048);
                for path in &files {
                    if let Ok(src) = fs::read_to_string(path) {
                        if let Ok(prog) = parse_saci_source(&src) {
                            if let Ok(art) = compile_saci_program(&prog, &mut allocator) {
                                if let Ok(loaded) =
                                    load_saci_binary(&mut saci_ckpt, &art.sacib_bytes)
                                {
                                    saci_descs.push(loaded.descriptor);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            dim,
            vocab_size,
            pure_model,
            ncm_gen,
            vq_dec,
            vq_tok_emb,
            saci_ckpt,
            saci_descs,
        })
    }
}

/// Output of one conversational turn (possibly spanning multiple 64-token chunks).
#[derive(Clone, Debug)]
pub struct TurnOutput {
    pub predicted_domain: usize,
    pub all_verified: bool,
    pub chunks_generated: usize,
    pub tokens: Vec<u16>,
    pub turn_index: u32,
}

/// Fixed-capacity KDA recurrent conversational session (`M in R^{dim x dim}`, bounded memory).
pub struct ConversationSession {
    pub dim: usize,
    pub encoder: SequenceEncoder,
    pub seq_scratch: SequenceScratch,
    /// Fixed-size KDA conversational memory matrix `[dim * dim]`.
    pub mem_state: Vec<f64>,
    pub mem_decay: f64,
    pub turn_count: u32,
    saci_scratch: [f64; 512],
    slot_buf: Vec<f64>,
    prompt_mat: Vec<f64>,
    ncm_feat: Vec<f64>,
}

impl ConversationSession {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            encoder: SequenceEncoder::new(dim),
            seq_scratch: SequenceScratch::new(dim),
            mem_state: vec![0.0_f64; dim * dim],
            mem_decay: 0.85,
            turn_count: 0,
            saci_scratch: [0.0_f64; 512],
            slot_buf: vec![0.0_f64; 8 * dim],
            prompt_mat: vec![0.0_f64; NCM_CHUNK_LEN * dim],
            ncm_feat: Vec::with_capacity(12 * dim),
        }
    }

    pub fn reset(&mut self) {
        self.mem_state.fill(0.0);
        self.turn_count = 0;
    }

    /// Update fixed-size KDA conversational memory with a unit vector `v` and return the memory readout.
    fn update_memory(&mut self, v: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let alpha = self.mem_decay;
        let mut readout = vec![0.0_f64; dim];
        for r in 0..dim {
            let row = r * dim;
            let mut s_k = 0.0_f64;
            for c in 0..dim {
                s_k += self.mem_state[row + c] * v[c];
            }
            let vr = v[r];
            let mut y_r = 0.0_f64;
            for c in 0..dim {
                let kc = v[c];
                let new_s = alpha * (self.mem_state[row + c] - s_k * kc) + vr * kc;
                self.mem_state[row + c] = new_s;
                y_r += new_s * kc;
            }
            readout[r] = y_r;
        }
        normalize_vec(&mut readout);
        readout
    }

    /// Run one conversational turn with up to `max_chunks` (each 64 tokens) of continuation.
    pub fn respond_turn(
        &mut self,
        model: &LoadedAdaModel,
        req_tokens: &[u16],
        num_table: &NumericVocabTable,
        max_chunks: usize,
    ) -> Result<TurnOutput, ErrorCode> {
        let dim = self.dim;
        let max_chunks = max_chunks.clamp(1, 8);

        // 1. Order-aware KDA sequence encoding + bag alignment for router compatibility
        let mut seq_vec = vec![0.0_f64; dim];
        self.encoder.encode_into(
            req_tokens,
            &model.vq_tok_emb,
            model.vocab_size,
            &mut self.seq_scratch,
            &mut seq_vec,
        )?;

        let mut bag_vec = vec![0.0_f64; dim];
        for &t in req_tokens {
            let tid = t as usize;
            if tid < model.vocab_size {
                let row = &model.vq_tok_emb[tid * dim..(tid + 1) * dim];
                for (dst, &src) in bag_vec.iter_mut().zip(row.iter()) {
                    *dst += src;
                }
            }
        }
        normalize_vec(&mut bag_vec);

        // Fuse current turn with prior conversational KDA memory (when turn_count > 0)
        let mem_read = self.update_memory(&seq_vec);
        let mut query_vec = vec![0.0_f64; dim];
        if self.turn_count == 0 {
            for d in 0..dim {
                query_vec[d] = 0.85 * bag_vec[d] + 0.15 * seq_vec[d];
            }
        } else {
            for d in 0..dim {
                query_vec[d] = 0.70 * bag_vec[d] + 0.15 * seq_vec[d] + 0.15 * mem_read[d];
            }
        }
        normalize_vec(&mut query_vec);

        // 2. 4-step Knowledge -> Reasoning -> SACI -> Simulation workspace pass
        let out_fw = forward_pure_weights_with_saci(
            &model.pure_model,
            &query_vec,
            req_tokens,
            num_table,
            &model.vq_tok_emb,
            model.vocab_size,
            &model.saci_ckpt,
            &model.saci_descs,
            &mut self.saci_scratch,
        );

        // 3. Multi-chunk NCM continuation loop
        let mut all_tokens: Vec<u16> = Vec::with_capacity(max_chunks * NCM_CHUNK_LEN);
        let mut rolling_ctx: Vec<u16> = req_tokens.to_vec();
        let mut prev_chunk: Vec<u16> = Vec::new();
        let mut chunks_generated = 0usize;

        for _chunk_idx in 0..max_chunks {
            self.slot_buf.fill(0.0);
            encode_token_slots(
                &rolling_ctx,
                &model.vq_tok_emb,
                model.vocab_size,
                dim,
                8,
                &mut self.slot_buf,
            )?;
            self.ncm_feat.clear();
            self.ncm_feat.extend_from_slice(&out_fw.ws_state);
            self.ncm_feat.extend_from_slice(&self.slot_buf);

            self.prompt_mat.fill(0.0);
            encode_prompt_matrix(
                &rolling_ctx,
                &model.vq_tok_emb,
                model.vocab_size,
                dim,
                NCM_CHUNK_LEN,
                &mut self.prompt_mat,
            )?;

            let z_hat = model.ncm_gen.generate_concept_chunk_with_prompt(
                &self.ncm_feat,
                &self.prompt_mat,
                out_fw.predicted_domain,
            );
            let chunk_toks = model.vq_dec.decode_chunk_parallel(&z_hat, NCM_CHUNK_LEN);
            chunks_generated += 1;

            // Stop on exact repetition plateau
            if !prev_chunk.is_empty() && chunk_toks == prev_chunk {
                break;
            }

            let mut hit_sep = false;
            for &t in &chunk_toks {
                if t == TOK_SEP {
                    hit_sep = true;
                    break;
                }
                if t != TOK_PAD {
                    all_tokens.push(t);
                }
            }

            // Update conversational KDA memory with emitted chunk
            let mut chunk_vec = vec![0.0_f64; dim];
            self.encoder.encode_into(
                &chunk_toks,
                &model.vq_tok_emb,
                model.vocab_size,
                &mut self.seq_scratch,
                &mut chunk_vec,
            )?;
            let _ = self.update_memory(&chunk_vec);

            if hit_sep {
                break;
            }

            // Roll context window (keep tail up to 64 tokens for next chunk)
            prev_chunk = chunk_toks.clone();
            rolling_ctx.extend_from_slice(&chunk_toks);
            if rolling_ctx.len() > NCM_CHUNK_LEN {
                let drop_n = rolling_ctx.len() - NCM_CHUNK_LEN;
                rolling_ctx.drain(0..drop_n);
            }
        }

        self.turn_count += 1;
        Ok(TurnOutput {
            predicted_domain: out_fw.predicted_domain,
            all_verified: out_fw.all_verified,
            chunks_generated,
            tokens: all_tokens,
            turn_index: self.turn_count,
        })
    }
}
