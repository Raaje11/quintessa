# Pending Work — Quintessa (Unified ADA Compiler + WSP Engine)

## Current Verified Baseline
- **Model:** `D:\TRACR\quintessa\output\ada_model.safetensors` (`32,340,472` bytes, `8,083,820` `F32` weights, `58` tensors)
- **Domain routing (`valid_acc`):** `3170/3170` (`100.00%`)
- **4-step verification (`all_4_verified`):** `3170/3170` (`100.00%`)
- **Compiled `Pipeline B` + `PonderNet` (`ponder_verified`):** `3170/3170` (`100.00%`)
- **Parallel NCM chunk cosine (`ncm_mean_cos`):** `0.7586`
- **Parallel NCM 64-token decoding (`ncm_parallel_tok_acc`):** `12931/16384` (`78.92%`)

## Completed Recently
- **Numeric Operand Extraction & Semantic Role Binding (`operand_grounding.rs`, `knowledge_base.rs`):**
  - `NumericVocabTable` + `extract_bound_operands` + `bind_operands_to_saci` wired into `forward_pure_weights_with_saci` so SACI circuits receive bound numeric operands and Workspace `s_state` receives both grounded role-values and SACI execution output.
- **Variable-Length 4-Head Content + Relative-Position Cross-Attention (`ncm_generator.rs`):**
  - Replaced static `ncm.a_copy [42, 64, 64]` with 4-head cross-attention (`ncm.w_q_attn [4, 128, 32]`, `ncm.w_k_heads [4, 128, 32]`, `ncm.rel_pos_bias [4, 127]`, `ncm.w_v_attn [128, 128]`).

---

## Pending Gaps to Reach 90%+ Decoding & Full Generalization

### 1. Per-Domain Specialization for Multi-Head Cross-Attention
- **Current state:** Removing per-domain `a_copy [42, 64, 64]` in favor of a single global 4-head cross-attention (`w_q_attn [4, 128, 32]`, `rel_pos_bias [4, 127]`) dropped `ncm_parallel_tok_acc` from `81.22% → 78.92%` because all 42 domains share the same 4 query heads and relative-position biases.
- **Required:** Per-domain relative-position biases (`[42, 4, 127]`, only `~85 KB`) and/or domain-conditioned query/value projections so Domain 7 and Domain 9 don't share one global alignment rule with static template domains.

### 2. NCM Capacity vs. Footprint Tuning
- **Current state:** `ncm.w_ncm_chunk` uses `exp_dim = 512` (`78.92%` / `0.7586` cosine at `32.3 MB` `F32`).
- **Required:** Re-tune `exp_dim` (`512 → 1024`) and `decoder_lambda` once per-domain cross-attention specialization is added.

### 3. Decoder Calibration & Multimodal Output Codecs (Deferred)
- **VQ decoder calibration:** Only investigate lightweight WSP `head_bias` calibration after `ncm_mean_cos > 0.90`.
- **Multimodal outputs (Documents, CAD, 3D, Images):** Currently only the `8,192`-vocab token codec (`vq_best.bpk`) is supported; non-text modalities require separate frozen VQ/codec packs.
