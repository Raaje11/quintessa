use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use quintessa::attention::{encode_prompt_matrix, encode_token_slots};
use quintessa::bpk_loader::load_bpk;
use quintessa::checkpoint::Checkpoint;
use quintessa::knowledge_base::{
    forward_pure_weights_with_saci, normalize_vec, PureWeightAdaModel, KB_DIM,
};
use quintessa::ncm_generator::{
    ParallelNcmGenerator, VqDecoderWeights, VqEncoderWeights, NCM_CHUNK_LEN,
};
use quintessa::operand_grounding::NumericVocabTable;
use quintessa::saci_compiler::{
    compile_saci_program, load_saci_binary, parse_saci_source, SaciCircuitDescriptor,
    SacisubstrateAllocator,
};
use quintessa::safetensors_writer::{read_safetensors, LoadedTensor};

const TOK_SUPERPOS: u16 = 10;
const TOK_THOUGHT: u16 = 11;
const TOK_SOLVER_START: u16 = 31;
const TOK_SOLVER_END: u16 = 72;
const TOK_BUDGET: u16 = 80;
const TOK_CLS: u16 = 101;
const TOK_SEP: u16 = 102;

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn find_tensor<'a>(tensors: &'a [LoadedTensor], name: &str) -> Result<&'a LoadedTensor, String> {
    tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("Missing tensor '{name}' in safetensors file"))
}

fn read_u16_stream(path: &Path) -> std::io::Result<Vec<u16>> {
    let mut f = File::open(path)?;
    let mut raw = Vec::new();
    f.read_to_end(&mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn featurize_tokens_vq(tokens: &[u16], tok_emb: &[f64], dim: usize, vocab: usize) -> Vec<f64> {
    let mut vec = vec![0.0_f64; dim];
    for &t in tokens {
        let tid = t as usize;
        if tid < vocab {
            let row = &tok_emb[tid * dim..(tid + 1) * dim];
            for (dst, &src) in vec.iter_mut().zip(row.iter()) {
                *dst += src;
            }
        }
    }
    normalize_vec(&mut vec);
    vec
}

#[derive(Default, Clone)]
struct DomainEvalStats {
    eps: usize,
    route_ok: usize,
    verified_ok: usize,
    cos_sum: f64,
    tok_ok: usize,
    tok_tot: usize,
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let model_path = PathBuf::from(
        parse_flag(args, "--model").unwrap_or_else(|| "output/ada_o1.safetensors".to_string()),
    );
    let valid_path = PathBuf::from(
        parse_flag(args, "--valid")
            .unwrap_or_else(|| r"D:\AdvancedADA\data_dal\valid.bin".to_string()),
    );
    let saci_dir = PathBuf::from(parse_flag(args, "--saci-dir").unwrap_or_else(|| {
        if Path::new("specs/saci").exists() {
            "specs/saci".to_string()
        } else {
            "quintessa/specs/saci".to_string()
        }
    }));
    let vq_path = PathBuf::from(
        parse_flag(args, "--vq")
            .unwrap_or_else(|| r"D:\AdvancedADA\models\ada_vq\vq_best.bpk".to_string()),
    );
    let vocab_path = Path::new(r"D:\AdvancedADA\data\vocab_ada_h1.txt");
    let vocab_names: Vec<String> = if vocab_path.exists() {
        fs::read_to_string(vocab_path)?
            .lines()
            .map(|l| l.trim().to_string())
            .collect()
    } else {
        Vec::new()
    };

    let tensors = read_safetensors(&model_path)?;
    let vq_tok_emb = find_tensor(&tensors, "vq.tok_emb.weight")?.data.clone();
    let vq_shape = &find_tensor(&tensors, "vq.tok_emb.weight")?.shape;
    let vocab_size = vq_shape[0];
    let dim = vq_shape[1];
    assert_eq!(dim, KB_DIM);

    let vh = &find_tensor(&tensors, "bridges.verify_halt")?.data;
    let pure_model = PureWeightAdaModel {
        w_gate: find_tensor(&tensors, "bridges.w_gate")?.data.clone(),
        b_gate: find_tensor(&tensors, "bridges.b_gate")?.data.clone(),
        w_knowledge: find_tensor(&tensors, "brain.kda.w_knowledge")?.data.clone(),
        w_reasoning: find_tensor(&tensors, "brain.mla.w_reasoning")?.data.clone(),
        w_saci_route: find_tensor(&tensors, "saci.router.w_route")?.data.clone(),
        w_saci_facts: find_tensor(&tensors, "saci.facts.w_facts")?.data.clone(),
        w_simulation: find_tensor(&tensors, "workspace.w_simulation")?
            .data
            .clone(),
        p_knowledge: find_tensor(&tensors, "brain.kda.p_knowledge")?.data.clone(),
        p_reasoning: find_tensor(&tensors, "brain.mla.p_reasoning")?.data.clone(),
        p_saci: find_tensor(&tensors, "saci.router.p_saci")?.data.clone(),
        p_simulation: find_tensor(&tensors, "workspace.p_simulation")?
            .data
            .clone(),
        verify_bias: vh[0],
        w_halt: [vh[1], vh[2]],
        b_halt: vh[3],
    };

    let dm_t = find_tensor(&tensors, "ncm.domain_means")?;
    let num_domains = dm_t.shape[0];
    let w_exp_t = find_tensor(&tensors, "ncm.w_expand")?;
    let exp_dim = w_exp_t.shape[1];
    let w_coarse_t = find_tensor(&tensors, "ncm.w_coarse")?;
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
        w_ncm_chunk: find_tensor(&tensors, "ncm.w_ncm_chunk")?.data.clone(),
        w_ncm_domain: Vec::new(),
        a_copy: Vec::new(),
        w_val: Vec::new(),
        coarse_len,
        w_coarse: w_coarse_t.data.clone(),
        w_q_attn: find_tensor(&tensors, "ncm.w_q_attn")?.data.clone(),
        w_k_heads: find_tensor(&tensors, "ncm.w_k_heads")?.data.clone(),
        rel_pos_bias: find_tensor(&tensors, "ncm.rel_pos_bias")?.data.clone(),
        w_v_attn: find_tensor(&tensors, "ncm.w_v_attn")?.data.clone(),
    };

    // Load VQ encoder & decoder from bpk for ground-truth latent cosine + 64-token decoding
    let bpk = load_bpk(&vq_path).map_err(|e| format!("Failed to load bpk: {e:?}"))?;
    let vq_enc = VqEncoderWeights::from_bpk(&bpk, dim)
        .map_err(|e| format!("Failed to load VqEncoderWeights from bpk: {e:?}"))?;
    let vq_dec = VqDecoderWeights::from_bpk(&bpk)
        .map_err(|e| format!("Failed to load VqDecoderWeights from bpk: {e:?}"))?;

    // Load SACI circuits from saci_dir
    let mut saci_files: Vec<PathBuf> = fs::read_dir(&saci_dir)?
        .filter_map(|e| e.ok().map(|de| de.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("saci"))
        .collect();
    saci_files.sort();
    let mut allocator = SacisubstrateAllocator::new(2048);
    let mut saci_ckpt = Checkpoint::new(Vec::new(), Vec::new()).unwrap();
    let mut saci_descs: Vec<SaciCircuitDescriptor> = Vec::with_capacity(saci_files.len());
    for path in &saci_files {
        let src = fs::read_to_string(path)?;
        let prog = parse_saci_source(&src)
            .map_err(|e| format!("SACI parse failed for {}: {e:?}", path.display()))?;
        let art = compile_saci_program(&prog, &mut allocator)
            .map_err(|e| format!("SACI compile failed for {}: {e:?}", path.display()))?;
        let loaded = load_saci_binary(&mut saci_ckpt, &art.sacib_bytes)
            .map_err(|e| format!("SACIB load failed for {}: {e:?}", path.display()))?;
        saci_descs.push(loaded.descriptor);
    }

    let num_table = NumericVocabTable::from_vocab_file(vocab_path, vocab_size);
    let valid_toks = read_u16_stream(&valid_path)?;

    // Parse all DAL episodes in valid.bin
    let mut episodes: Vec<(usize, Vec<u16>, Vec<u16>)> = Vec::new();
    let mut i = 0usize;
    while i < valid_toks.len() {
        if valid_toks[i] != TOK_CLS {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut end = start;
        while end < valid_toks.len() && valid_toks[end] != TOK_SEP && valid_toks[end] != TOK_CLS {
            end += 1;
        }
        let ep = &valid_toks[start..end];
        i = if end < valid_toks.len() && valid_toks[end] == TOK_SEP {
            end + 1
        } else {
            end
        };
        if ep.is_empty() || ep[0] != TOK_SUPERPOS {
            continue;
        }
        let budget_pos = ep.iter().position(|&t| t == TOK_BUDGET).unwrap_or(0);
        let req_start = (budget_pos + 5).min(ep.len());
        let thought_pos = ep
            .iter()
            .position(|&t| t == TOK_THOUGHT)
            .unwrap_or(ep.len());
        let req_end = thought_pos.max(req_start);
        if req_end <= req_start || thought_pos >= ep.len() {
            continue;
        }
        let mut dom = 0usize;
        for &t in ep {
            if (TOK_SOLVER_START..=TOK_SOLVER_END).contains(&t) {
                dom = (t - TOK_SOLVER_START) as usize;
                break;
            }
        }
        let req_toks = ep[req_start..req_end].to_vec();
        let out_slice = &ep[thought_pos..ep.len().min(thought_pos + NCM_CHUNK_LEN)];
        let mut target_toks = vec![0u16; NCM_CHUNK_LEN];
        target_toks[..out_slice.len()].copy_from_slice(out_slice);
        episodes.push((dom, req_toks, target_toks));
    }

    // Parallel evaluation across threads
    let n_eps = episodes.len();
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_eps.max(1));
    let chunk_sz = n_eps.div_ceil(n_threads);

    let partials: Vec<Vec<DomainEvalStats>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t_idx in 0..n_threads {
            let start = t_idx * chunk_sz;
            if start >= n_eps {
                break;
            }
            let end = (start + chunk_sz).min(n_eps);
            let eps_ref = &episodes[start..end];
            let pm_ref = &pure_model;
            let ncm_ref = &ncm_gen;
            let enc_ref = &vq_enc;
            let dec_ref = &vq_dec;
            let emb_ref = &vq_tok_emb;
            let num_ref = &num_table;
            let ckpt_ref = &saci_ckpt;
            let desc_ref = &saci_descs;
            handles.push(s.spawn(move || {
                let mut local = vec![DomainEvalStats::default(); num_domains];
                let mut saci_scratch = [0.0_f64; 512];
                let mut slot_buf = vec![0.0_f64; 8 * dim];
                let mut prompt_mat = vec![0.0_f64; NCM_CHUNK_LEN * dim];
                let mut ncm_feat = Vec::with_capacity(12 * dim);

                for (gt_dom, req_toks, target_toks) in eps_ref {
                    let req_vec = featurize_tokens_vq(req_toks, emb_ref, dim, vocab_size);
                    let out_fw = forward_pure_weights_with_saci(
                        pm_ref,
                        &req_vec,
                        req_toks,
                        num_ref,
                        emb_ref,
                        vocab_size,
                        ckpt_ref,
                        desc_ref,
                        &mut saci_scratch,
                    );

                    slot_buf.fill(0.0);
                    let _ =
                        encode_token_slots(req_toks, emb_ref, vocab_size, dim, 8, &mut slot_buf);
                    ncm_feat.clear();
                    ncm_feat.extend_from_slice(&out_fw.ws_state);
                    ncm_feat.extend_from_slice(&slot_buf);

                    prompt_mat.fill(0.0);
                    let _ = encode_prompt_matrix(
                        req_toks,
                        emb_ref,
                        vocab_size,
                        dim,
                        NCM_CHUNK_LEN,
                        &mut prompt_mat,
                    );

                    let z_hat = ncm_ref.generate_concept_chunk_with_prompt(
                        &ncm_feat,
                        &prompt_mat,
                        out_fw.predicted_domain,
                    );
                    let tgt = enc_ref.encode_chunk(target_toks, emb_ref, vocab_size);
                    let dot: f64 = z_hat.iter().zip(tgt.iter()).map(|(a, b)| a * b).sum();
                    let nz = z_hat.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let nt = tgt.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let cos = if nz > 1e-12 && nt > 1e-12 {
                        dot / (nz * nt)
                    } else {
                        0.0
                    };

                    let pred_toks = dec_ref.decode_chunk_parallel(&z_hat, NCM_CHUNK_LEN);
                    let tok_ok = pred_toks
                        .iter()
                        .zip(target_toks.iter())
                        .filter(|(p, t)| p == t)
                        .count();

                    let st = &mut local[*gt_dom];
                    st.eps += 1;
                    if out_fw.predicted_domain == *gt_dom {
                        st.route_ok += 1;
                    }
                    if out_fw.all_verified {
                        st.verified_ok += 1;
                    }
                    st.cos_sum += cos;
                    st.tok_ok += tok_ok;
                    st.tok_tot += NCM_CHUNK_LEN;
                }
                local
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut stats = vec![DomainEvalStats::default(); num_domains];
    for p in partials {
        for (dst, src) in stats.iter_mut().zip(p.into_iter()) {
            dst.eps += src.eps;
            dst.route_ok += src.route_ok;
            dst.verified_ok += src.verified_ok;
            dst.cos_sum += src.cos_sum;
            dst.tok_ok += src.tok_ok;
            dst.tok_tot += src.tok_tot;
        }
    }

    println!(
        "Per-Domain Evaluation on {} ({} DAL episodes, elapsed={} ms):",
        valid_path.display(),
        n_eps,
        t0.elapsed().as_millis()
    );
    println!(
        "DOM | TOKEN              |   EPS | ROUTE_ACC | VERIFY_4 | NCM_COS | TOK_ACC (MATCH/TOT)"
    );
    println!("----+--------------------+-------+-----------+----------+---------+-------------------------");
    let mut tot_eps = 0usize;
    let mut tot_route = 0usize;
    let mut tot_ver = 0usize;
    let mut tot_cos = 0.0_f64;
    let mut tot_tok_ok = 0usize;
    let mut tot_tok_tot = 0usize;

    for (d, st) in stats.iter().enumerate() {
        let name = if d == 0 {
            "BASE/CONV"
        } else {
            let tid = (TOK_SOLVER_START as usize) + d - 1;
            vocab_names.get(tid).map(|s| s.as_str()).unwrap_or("SOLVER")
        };
        if st.eps == 0 {
            println!("{d:3} | {name:<18} |     0 |       n/a |      n/a |     n/a | n/a");
            continue;
        }
        tot_eps += st.eps;
        tot_route += st.route_ok;
        tot_ver += st.verified_ok;
        tot_cos += st.cos_sum;
        tot_tok_ok += st.tok_ok;
        tot_tok_tot += st.tok_tot;
        println!(
            "{d:3} | {name:<18} | {:5} | {:8.2}% | {:7.2}% |  {:6.4} | {:6.2}% ({}/{})",
            st.eps,
            100.0 * (st.route_ok as f64) / (st.eps as f64),
            100.0 * (st.verified_ok as f64) / (st.eps as f64),
            st.cos_sum / (st.eps as f64),
            100.0 * (st.tok_ok as f64) / (st.tok_tot as f64),
            st.tok_ok,
            st.tok_tot,
        );
    }
    println!("----+--------------------+-------+-----------+----------+---------+-------------------------");
    println!(
        "ALL | TOTAL              | {:5} | {:8.2}% | {:7.2}% |  {:6.4} | {:6.2}% ({}/{})",
        tot_eps,
        100.0 * (tot_route as f64) / (tot_eps.max(1) as f64),
        100.0 * (tot_ver as f64) / (tot_eps.max(1) as f64),
        tot_cos / (tot_eps.max(1) as f64),
        100.0 * (tot_tok_ok as f64) / (tot_tok_tot.max(1) as f64),
        tot_tok_ok,
        tot_tok_tot,
    );

    Ok(())
}
