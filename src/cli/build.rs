//! Unified Quintessa Compiler + WSP Writer + Verifier.
//!
//! 100% input-driven: reads `--spec <spec>`, `--train <train.bin>`, `--valid <valid.bin>`,
//! `--saci-dir <dir_of_saci_files>`, optional `--vq <vq.bpk>`, and `--out <model.safetensors>`.
//! Runs ADA architecture compilation, SACI circuit compilation, WSP synthesis, and verification
//! in-memory and emits ONE AND ONLY ONE output file: a verified pure-weight `.safetensors` model.

use std::fs::{self, File};
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use quintessa::attention::{encode_prompt_matrix, encode_token_sequence, encode_token_slots};
use quintessa::capability::{
    CapabilityId, CapabilitySpec, CapabilityType, ObjectiveKind, SemanticsKind, TrainingObjective,
    ValueType, VerificationSpec, MAX_CAP_DEPS, MAX_CAP_IO,
};
use quintessa::ir::{
    ArchitectureSpec, KdaConfig, MlaConfig, NcmArtifactKind, NcmOutputIr, SaciIr, SaciOpSpec,
    WorkflowPlanIr, WorkspaceIr,
};
use quintessa::runtime::{
    compile_domain_architecture, execute_query_pipeline_b, AdaQueryScratch, CompiledAdaModel,
};
use quintessa::sys::{
    ArrayVec, AttentionKind, Backend, ErrorCode, NcmNodeId, NumericalMode, SaciOpId, TelemetryBus,
    WorkspaceState,
};
use quintessa::wsp::verify_exclusivity;

use quintessa::bpk_loader::load_bpk;
use quintessa::checkpoint::Checkpoint;
use quintessa::knowledge_base::{
    build_wsp_knowledge_base, compile_pure_weight_model, compute_exclusive_neuron_masks,
    forward_pure_weights_with_saci, normalize_vec, KbSample, KB_DIM,
};
use quintessa::ncm_generator::{
    compile_parallel_ncm_wsp_copy_and_residual, VqDecoderWeights, VqEncoderWeights, NCM_CHUNK_LEN,
};
use quintessa::operand_grounding::NumericVocabTable;
use quintessa::saci_compiler::{
    compile_saci_program, load_saci_binary, parse_saci_source, SaciCircuitDescriptor,
    SacisubstrateAllocator,
};
use quintessa::safetensors_writer::{
    write_mixed_safetensors, write_safetensors, NamedTensor, TypedNamedTensor, WeightDtype,
};
use quintessa::sys::TensorId;
use quintessa::token_decoder::{token_embedding, EMB_DIM};

const TOK_SUPERPOS: u16 = 10;
const TOK_THOUGHT: u16 = 11;
const TOK_FACTS: u16 = 16;
const TOK_SOLVER_START: u16 = 31;
const TOK_SOLVER_END: u16 = 72;
const TOK_ROUTE: u16 = 79;
const TOK_BUDGET: u16 = 80;
const TOK_CLS: u16 = 101;
const TOK_SEP: u16 = 102;

struct ParsedSpecSheet {
    arch: ArchitectureSpec,
    topology: Vec<AttentionKind>,
    neurons_per_capability: usize,
    decoder_lambda: f64,
}

fn parse_spec_sheet(path: &Path) -> Result<ParsedSpecSheet, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    let mut num_neurons = 256usize;
    let mut substrate_dim = KB_DIM;
    let mut tile_size = 16usize;
    let mut vq_codebook_size = 8192u32;
    let mut topology = vec![AttentionKind::Kda; 6];
    topology.push(AttentionKind::Mla);
    topology.push(AttentionKind::Mla);

    let mut kda_chunk = 64u16;
    let mut kda_rank = 4u16;
    let mut kda_state = 128u16;
    let mut mla_latent = 64u16;
    let mut mla_heads = 8u16;
    let mut mla_head_dim = 64u16;
    let mut ws_slots = 16u16;
    let mut ws_iters = 8u16;
    let mut ws_tol = 1e-6_f64;
    let mut neurons_per_cap = 64usize;
    let mut dec_lambda = 0.1_f64;

    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            let val = v.trim();
            match key {
                "num_neurons" => num_neurons = val.parse()?,
                "substrate_dim" => substrate_dim = val.parse()?,
                "tile_size" => tile_size = val.parse()?,
                "vq_codebook_size" => vq_codebook_size = val.parse()?,
                "topology" => {
                    topology.clear();
                    for tok in val.split(',') {
                        match tok.trim().to_ascii_uppercase().as_str() {
                            "KDA" => topology.push(AttentionKind::Kda),
                            "MLA" => topology.push(AttentionKind::Mla),
                            other => {
                                return Err(format!("Unknown AttentionKind in spec: {other}").into())
                            }
                        }
                    }
                }
                "kda_chunk_size" => kda_chunk = val.parse()?,
                "kda_dplr_rank" => kda_rank = val.parse()?,
                "kda_state_dim" => kda_state = val.parse()?,
                "mla_kv_latent_dim" => mla_latent = val.parse()?,
                "mla_num_heads" => mla_heads = val.parse()?,
                "mla_head_dim" => mla_head_dim = val.parse()?,
                "workspace_slots" => ws_slots = val.parse()?,
                "workspace_max_iters" => ws_iters = val.parse()?,
                "workspace_halting_tolerance" => ws_tol = val.parse()?,
                "neurons_per_capability" => neurons_per_cap = val.parse()?,
                "decoder_lambda" => dec_lambda = val.parse()?,
                _ => {}
            }
        }
    }

    Ok(ParsedSpecSheet {
        arch: ArchitectureSpec {
            vq_codebook_size,
            num_neurons,
            substrate_dim,
            tile_size,
            kda: KdaConfig {
                chunk_size: kda_chunk,
                dplr_rank: kda_rank,
                state_dim: kda_state,
            },
            mla: MlaConfig {
                kv_latent_dim: mla_latent,
                num_heads: mla_heads,
                head_dim: mla_head_dim,
            },
            workspace: WorkspaceIr {
                scratchpad_slots: ws_slots,
                max_iters: ws_iters,
                halting_tolerance: ws_tol,
            },
        },
        topology,
        neurons_per_capability: neurons_per_cap,
        decoder_lambda: dec_lambda,
    })
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

#[allow(dead_code)]
fn featurize_sequence_substrate(
    model: &CompiledAdaModel,
    scratch: &mut AdaQueryScratch,
    tokens: &[u16],
    tok_emb: &[f64],
    dim: usize,
    vocab: usize,
) -> Vec<f64> {
    let mut vec = vec![0.0_f64; dim];
    if encode_token_sequence(
        &model.ir.attention,
        tokens,
        tok_emb,
        vocab,
        dim,
        &mut vec,
        &mut scratch.attn,
    )
    .is_err()
    {
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
    }
    vec
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

fn extract_kb_samples(tokens: &[u16], tok_emb: &[f64], dim: usize, vocab: usize) -> Vec<KbSample> {
    let mut samples = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != TOK_CLS {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut end = start;
        while end < tokens.len() && tokens[end] != TOK_SEP && tokens[end] != TOK_CLS {
            end += 1;
        }
        let ep = &tokens[start..end];
        i = if end < tokens.len() && tokens[end] == TOK_SEP {
            end + 1
        } else {
            end
        };
        if ep.is_empty() {
            continue;
        }

        if ep[0] == TOK_SUPERPOS {
            let mut solver_id: Option<usize> = None;
            for w in ep.windows(3) {
                if w[0] == TOK_ROUTE && w[2] >= TOK_SOLVER_START && w[2] <= TOK_SOLVER_END {
                    solver_id = Some((w[2] - TOK_SOLVER_START) as usize);
                    break;
                }
            }
            let Some(sid) = solver_id else {
                continue;
            };
            let budget_pos = ep.iter().position(|&t| t == TOK_BUDGET).unwrap_or(0);
            let req_start = (budget_pos + 5).min(ep.len());
            let thought_pos = ep
                .iter()
                .position(|&t| t == TOK_THOUGHT)
                .unwrap_or(ep.len());
            let facts_pos = ep.iter().position(|&t| t == TOK_FACTS).unwrap_or(ep.len());

            let req_end = thought_pos.max(req_start);
            let req_toks = &ep[req_start..req_end];
            if req_toks.is_empty() {
                continue;
            }
            let thought_toks = if thought_pos < facts_pos {
                &ep[thought_pos..facts_pos]
            } else {
                req_toks
            };
            let facts_toks = if facts_pos < ep.len() {
                &ep[facts_pos..]
            } else {
                thought_toks
            };

            samples.push(KbSample {
                domain_id: sid,
                protected: false,
                req_vec: featurize_tokens_vq(req_toks, tok_emb, dim, vocab),
                thought_vec: featurize_tokens_vq(thought_toks, tok_emb, dim, vocab),
                facts_vec: featurize_tokens_vq(facts_toks, tok_emb, dim, vocab),
            });
        } else {
            let prompt_len = ep.len().min(32);
            let pv = featurize_tokens_vq(&ep[..prompt_len], tok_emb, dim, vocab);
            samples.push(KbSample {
                domain_id: 0,
                protected: true,
                req_vec: pv.clone(),
                thought_vec: pv.clone(),
                facts_vec: pv,
            });
        }
    }

    samples
}

fn io_vec(vals: &[ValueType]) -> ArrayVec<ValueType, MAX_CAP_IO> {
    let mut v = ArrayVec::new();
    for &x in vals {
        assert_eq!(v.push(x), ErrorCode::None);
    }
    v
}

fn dep_vec(vals: &[CapabilityId]) -> ArrayVec<CapabilityId, MAX_CAP_DEPS> {
    let mut v = ArrayVec::new();
    for &x in vals {
        assert_eq!(v.push(x), ErrorCode::None);
    }
    v
}

fn parse_flag(args: &[String], flag: &str) -> Option<PathBuf> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| PathBuf::from(&w[1]))
}

fn parse_str_flag(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ModelKind {
    AdaO1 = 0,
    CustomSpec = 1,
    CompactEdge = 2,
}

impl ModelKind {
    fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "2" | "custom" | "custom-spec" => ModelKind::CustomSpec,
            "3" | "compact" | "compact-edge" => ModelKind::CompactEdge,
            _ => ModelKind::AdaO1,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ModelKind::AdaO1 => "ada-o1",
            ModelKind::CustomSpec => "custom-spec",
            ModelKind::CompactEdge => "compact-edge",
        }
    }
}

struct BuildFormConfig {
    model_kind: ModelKind,
    etf_features: Option<usize>,
    substrate_dim_override: Option<usize>,
    spec_path: PathBuf,
    train_path: PathBuf,
    valid_path: PathBuf,
    saci_dir: PathBuf,
    vq_path: Option<PathBuf>,
    out_path: PathBuf,
    dtype: WeightDtype,
}

fn prompt_field(reader: &mut impl BufRead, label: &str, default: &str) -> io::Result<String> {
    if default.is_empty() {
        print!("  {label}: ");
    } else {
        print!("  {label} [{default}]: ");
    }
    io::stdout().flush()?;
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn run_interactive_form(args: &[String]) -> Result<BuildFormConfig, Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    println!("=== ADA Compiler + WSP Writer :: Pure-Weights Model Builder Form ===");
    println!("Select model architecture:");
    println!("  [1] ada-o1        (KDAx6 + MLAx2, Parallel NCM + SACI circuits)");
    println!("  [2] custom-spec   (Use topology & dims from uploaded .spec file)");
    println!("  [3] compact-edge  (KDAx4 + MLAx1, compact edge footprint)");

    let def_kind = parse_str_flag(args, "--model-kind").unwrap_or_else(|| "1".to_string());
    let kind_in = prompt_field(&mut reader, "Model kind (1/2/3 or name)", &def_kind)?;
    let model_kind = ModelKind::from_str(&kind_in);

    let def_etf = parse_str_flag(args, "--etf-features").unwrap_or_else(|| "1024".to_string());
    let etf_in = prompt_field(
        &mut reader,
        "ETF features number (4 families/neuron, e.g. 1024 -> 256 neurons)",
        &def_etf,
    )?;
    let etf_features = etf_in.parse::<usize>().ok().filter(|&n| n > 0);

    let dim_in = prompt_field(
        &mut reader,
        "Substrate dimension (blank = use spec default)",
        "",
    )?;
    let substrate_dim_override = if dim_in.is_empty() {
        None
    } else {
        Some(dim_in.parse::<usize>()?)
    };

    println!("\nUploaded source files required to compile & convert into feature weights:");
    let def_spec = parse_str_flag(args, "--spec").unwrap_or_else(|| {
        if Path::new("specs/ada_o1.spec").exists() {
            "specs/ada_o1.spec"
        } else {
            "quintessa/specs/ada_o1.spec"
        }
        .to_string()
    });
    let spec_path = PathBuf::from(prompt_field(&mut reader, "Spec sheet (.spec)", &def_spec)?);

    let def_train = parse_str_flag(args, "--train")
        .unwrap_or_else(|| r"D:\AdvancedADA\data_dal\train.bin".to_string());
    let train_path = PathBuf::from(prompt_field(
        &mut reader,
        "Training dataset (train.bin)",
        &def_train,
    )?);

    let def_valid = parse_str_flag(args, "--valid")
        .unwrap_or_else(|| r"D:\AdvancedADA\data_dal\valid.bin".to_string());
    let valid_path = PathBuf::from(prompt_field(
        &mut reader,
        "Validation dataset (valid.bin)",
        &def_valid,
    )?);

    let def_saci = parse_str_flag(args, "--saci-dir").unwrap_or_else(|| {
        if Path::new("specs/saci").exists() {
            "specs/saci"
        } else {
            "quintessa/specs/saci"
        }
        .to_string()
    });
    let saci_dir = PathBuf::from(prompt_field(
        &mut reader,
        "SACI algorithms directory (*.saci)",
        &def_saci,
    )?);

    let def_vq = parse_str_flag(args, "--vq")
        .unwrap_or_else(|| r"D:\AdvancedADA\models\ada_vq\vq_best.bpk".to_string());
    let vq_in = prompt_field(
        &mut reader,
        "VQ / embeddings BurnPack (.bpk, or 'none' for ETF-only)",
        &def_vq,
    )?;
    let vq_path = if vq_in.is_empty() || vq_in.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(PathBuf::from(vq_in))
    };

    println!("\nPure-weights export settings:");
    let def_out = parse_str_flag(args, "--out").unwrap_or_else(|| {
        if Path::new("specs/ada_o1.spec").exists() {
            "output/ada_model.safetensors"
        } else {
            "quintessa/output/ada_model.safetensors"
        }
        .to_string()
    });
    let out_path = PathBuf::from(prompt_field(
        &mut reader,
        "Output pure-weights path (.safetensors)",
        &def_out,
    )?);

    let def_dtype = parse_str_flag(args, "--dtype").unwrap_or_else(|| "f64".to_string());
    let dtype_in = prompt_field(&mut reader, "Weight dtype (f64 / f32)", &def_dtype)?;
    let dtype = if dtype_in.eq_ignore_ascii_case("f32") {
        WeightDtype::F32
    } else {
        WeightDtype::F64
    };
    println!();

    Ok(BuildFormConfig {
        model_kind,
        etf_features,
        substrate_dim_override,
        spec_path,
        train_path,
        valid_path,
        saci_dir,
        vq_path,
        out_path,
        dtype,
    })
}

pub fn run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let use_form = has_flag(args, "--form")
        || has_flag(args, "--interactive")
        || parse_flag(args, "--spec").is_none()
        || parse_flag(args, "--train").is_none()
        || parse_flag(args, "--valid").is_none()
        || parse_flag(args, "--saci-dir").is_none();

    let cfg = if use_form {
        run_interactive_form(args)?
    } else {
        let model_kind = parse_str_flag(args, "--model-kind")
            .map(|s| ModelKind::from_str(&s))
            .unwrap_or(ModelKind::AdaO1);
        let etf_features = parse_str_flag(args, "--etf-features")
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0);
        let dtype = match parse_str_flag(args, "--dtype")
            .as_deref()
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("f32") => WeightDtype::F32,
            Some("bf16") | Some("mixed") => WeightDtype::BF16,
            Some("f16") => WeightDtype::F16,
            _ => WeightDtype::F64,
        };
        BuildFormConfig {
            model_kind,
            etf_features,
            substrate_dim_override: None,
            spec_path: parse_flag(args, "--spec").unwrap(),
            train_path: parse_flag(args, "--train").unwrap(),
            valid_path: parse_flag(args, "--valid").unwrap(),
            saci_dir: parse_flag(args, "--saci-dir").unwrap(),
            vq_path: parse_flag(args, "--vq"),
            out_path: parse_flag(args, "--out")
                .unwrap_or_else(|| PathBuf::from("output/ada_model.safetensors")),
            dtype,
        }
    };

    let spec_path = cfg.spec_path;
    let train_path = cfg.train_path;
    let valid_path = cfg.valid_path;
    let saci_dir = cfg.saci_dir;
    let vq_path = cfg.vq_path;
    let out_path = cfg.out_path;
    if out_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(format!(
            "Output path must strictly end with .safetensors (got: {})",
            out_path.display()
        )
        .into());
    }

    // 1. Parse external Architecture Spec Sheet & apply form/CLI overrides
    let mut parsed_spec = parse_spec_sheet(&spec_path)?;
    if let Some(etf_n) = cfg.etf_features {
        parsed_spec.arch.num_neurons = etf_n.div_ceil(4).max(1);
    }
    if let Some(d_ovr) = cfg.substrate_dim_override {
        parsed_spec.arch.substrate_dim = d_ovr;
    }
    if cfg.model_kind == ModelKind::CompactEdge {
        parsed_spec.topology = vec![
            AttentionKind::Kda,
            AttentionKind::Kda,
            AttentionKind::Kda,
            AttentionKind::Kda,
            AttentionKind::Mla,
        ];
    }
    let arch_spec = parsed_spec.arch;
    let dim = arch_spec.substrate_dim;
    let vocab_size = arch_spec.vq_codebook_size as usize;

    // 2. Discover & compile all external `*.saci` files from `--saci-dir`
    let mut saci_files: Vec<PathBuf> = fs::read_dir(&saci_dir)?
        .filter_map(|e| e.ok().map(|de| de.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("saci"))
        .collect();
    saci_files.sort();
    if saci_files.is_empty() {
        return Err(format!("No .saci files found in {}", saci_dir.display()).into());
    }

    let mut allocator = SacisubstrateAllocator::new(2048);
    let mut saci_ckpt = Checkpoint::new(Vec::new(), Vec::new()).unwrap();
    let mut saci_descs: Vec<SaciCircuitDescriptor> = Vec::with_capacity(saci_files.len());
    let mut saci_ops: Vec<SaciOpSpec> = Vec::with_capacity(saci_files.len());

    for path in &saci_files {
        let src = fs::read_to_string(path)?;
        let prog = parse_saci_source(&src)
            .map_err(|e| format!("SACI parse failed for {}: {e:?}", path.display()))?;
        let art = compile_saci_program(&prog, &mut allocator)
            .map_err(|e| format!("SACI compile failed for {}: {e:?}", path.display()))?;
        let loaded = load_saci_binary(&mut saci_ckpt, &art.sacib_bytes)
            .map_err(|e| format!("SACI load failed for {}: {e:?}", path.display()))?;
        saci_ops.push(SaciOpSpec {
            id: SaciOpId(loaded.descriptor.circuit_id as u16),
            numerical_mode: NumericalMode::Exact,
            tolerance: loaded.descriptor.absolute_tolerance,
            backend: Backend::Cpu,
            precondition_flags: 0x01,
            postcondition_flags: 0x01,
        });
        saci_descs.push(loaded.descriptor);
    }

    // 3. Compile ADA Architecture via `ada-compiler`
    let first_saci_id = saci_ops[0].id;
    let caps = [
        CapabilitySpec {
            id: CapabilityId(1),
            cap_type: CapabilityType::Knowledge,
            inputs: io_vec(&[ValueType::Vector]),
            outputs: io_vec(&[ValueType::Vector]),
            semantics: SemanticsKind::ConceptRelation,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: dep_vec(&[]),
            training_objective: Some(TrainingObjective {
                kind: ObjectiveKind::ContrastiveSemantic,
                constraint_weight: 0.5,
                target_tolerance: 1e-3,
            }),
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-3,
                requires_formal_proof: false,
            },
            saci_binding: None,
        },
        CapabilitySpec {
            id: CapabilityId(2),
            cap_type: CapabilityType::Reasoning,
            inputs: io_vec(&[ValueType::Vector]),
            outputs: io_vec(&[ValueType::Int]),
            semantics: SemanticsKind::InferencePolicy,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: dep_vec(&[CapabilityId(1)]),
            training_objective: Some(TrainingObjective {
                kind: ObjectiveKind::TracePolicyImitation,
                constraint_weight: 0.25,
                target_tolerance: 1e-3,
            }),
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-3,
                requires_formal_proof: false,
            },
            saci_binding: Some(first_saci_id),
        },
        CapabilitySpec {
            id: CapabilityId(3),
            cap_type: CapabilityType::Computation,
            inputs: io_vec(&[ValueType::Vector]),
            outputs: io_vec(&[ValueType::Vector]),
            semantics: SemanticsKind::DeterministicFormula,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: dep_vec(&[CapabilityId(2)]),
            training_objective: None,
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-9,
                requires_formal_proof: true,
            },
            saci_binding: Some(first_saci_id),
        },
        CapabilitySpec {
            id: CapabilityId(4),
            cap_type: CapabilityType::Simulation,
            inputs: io_vec(&[ValueType::State, ValueType::Vector]),
            outputs: io_vec(&[ValueType::State]),
            semantics: SemanticsKind::StateTransition,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: dep_vec(&[CapabilityId(3)]),
            training_objective: Some(TrainingObjective {
                kind: ObjectiveKind::PhysicsConstrainedTrajectory,
                constraint_weight: 1.0,
                target_tolerance: 1e-4,
            }),
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-4,
                requires_formal_proof: false,
            },
            saci_binding: None,
        },
        CapabilitySpec {
            id: CapabilityId(5),
            cap_type: CapabilityType::Verification,
            inputs: io_vec(&[ValueType::State]),
            outputs: io_vec(&[ValueType::Bool]),
            semantics: SemanticsKind::InvariantPredicate,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: dep_vec(&[CapabilityId(4)]),
            training_objective: None,
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-9,
                requires_formal_proof: true,
            },
            saci_binding: None,
        },
    ];
    let mut ada_tel: TelemetryBus<256> = TelemetryBus::new();
    let (ada_model, _graph, ada_manifest) = compile_domain_architecture(
        &arch_spec,
        &parsed_spec.topology,
        &caps,
        SaciIr::new(saci_ops),
        parsed_spec.neurons_per_capability,
        &mut ada_tel,
    )
    .map_err(|e| format!("ADA compile failed: {e:?}"))?;

    // 4. Load external `--vq` BurnPack (`tok_emb`, `head_bias`, `ln_dec`, `dec1`, `dec2` + encoder)
    let (vq_tok_emb, vq_head_bias, vq_dec, vq_enc) = if let Some(ref vp) = vq_path {
        let vq_archive = load_bpk(vp).map_err(|e| format!("Failed to load VQ .bpk: {e:?}"))?;
        let dec = VqDecoderWeights::from_bpk(&vq_archive)
            .map_err(|e| format!("Failed to load VQ decoder blocks: {e:?}"))?;
        let enc = VqEncoderWeights::from_bpk(&vq_archive, dec.dim)
            .map_err(|e| format!("Failed to load VQ encoder blocks: {e:?}"))?;
        (
            dec.tok_emb.clone(),
            dec.head_bias.clone(),
            Some(dec),
            Some(enc),
        )
    } else {
        let mut te = vec![0.0_f64; vocab_size * dim];
        let mut e32 = [0.0_f64; EMB_DIM];
        for tid in 0..vocab_size {
            token_embedding(tid as u16, &mut e32);
            for d in 0..dim {
                te[tid * dim + d] = e32[d % EMB_DIM];
            }
        }
        (te, vec![0.0_f64; vocab_size], None, None)
    };

    // 5. Load external `--train` & `--valid` datasets and WSP-synthesize 4-step weights
    let vocab_txt_path = Path::new(r"D:\AdvancedADA\models\ada_o1\vocab.txt");
    let num_table = if vocab_txt_path.exists() {
        NumericVocabTable::from_vocab_file(vocab_txt_path, vocab_size)
    } else {
        NumericVocabTable::default_ada_o1(vocab_size)
    };
    let mut ada_scratch = AdaQueryScratch::new(&ada_model, 8);
    let mut train_toks = read_u16_stream(&train_path)?;
    if let Some(aug_n) = parse_str_flag(args, "--augment-conv")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        if vocab_txt_path.exists() {
            if let Ok(tok) =
                quintessa::sequence_encoder::VocabTokenizer::from_vocab_file(vocab_txt_path)
            {
                let eps = quintessa::semantic_data::generate_semantic_corpus(&tok, aug_n, 2026);
                let mut aug_stream = quintessa::semantic_data::encode_episodes_to_u16_stream(&eps);
                // Prepend so bounded NCM episode extraction always includes conversational/semantic data
                aug_stream.extend_from_slice(&train_toks);
                train_toks = aug_stream;
            }
        }
    }
    let valid_toks = read_u16_stream(&valid_path)?;
    let train_samples = extract_kb_samples(&train_toks, &vq_tok_emb, dim, vocab_size);
    let valid_samples = extract_kb_samples(&valid_toks, &vq_tok_emb, dim, vocab_size);

    let kb = build_wsp_knowledge_base(&train_samples)
        .map_err(|e| format!("WSP KB synthesis failed: {e:?}"))?;
    drop(train_samples);
    let pure_model = compile_pure_weight_model(&kb);

    let valid_dal: Vec<&KbSample> = valid_samples.iter().filter(|s| !s.protected).collect();
    let mut pw_correct = 0usize;
    let mut pw_verified = 0usize;
    let mut saci_exec_scratch = vec![0.0_f64; 512];
    for s in &valid_dal {
        let out = forward_pure_weights_with_saci(
            &pure_model,
            &s.req_vec,
            &[],
            &num_table,
            &vq_tok_emb,
            vocab_size,
            &saci_ckpt,
            &saci_descs,
            &mut saci_exec_scratch,
        );
        if out.predicted_domain == s.domain_id {
            pw_correct += 1;
        }
        if out.all_verified {
            pw_verified += 1;
        }
    }

    // 6. WSP-synthesize Unblurred Cross-Position Copy + Compact ReLU-ETF Parallel NCM Generator
    let extract_ncm_pairs = |tokens: &[u16],
                             max_eps: usize|
     -> (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<usize>,
        Vec<Vec<u16>>,
    ) {
        let mut ws_list = Vec::new();
        let mut prompt_list = Vec::new();
        let mut dom_list = Vec::new();
        let mut tok_chunks = Vec::new();
        let mut i = 0usize;
        while i < tokens.len() && ws_list.len() < max_eps {
            if tokens[i] != TOK_CLS {
                i += 1;
                continue;
            }
            let start = i + 1;
            let mut end = start;
            while end < tokens.len() && tokens[end] != TOK_SEP && tokens[end] != TOK_CLS {
                end += 1;
            }
            let ep = &tokens[start..end];
            i = if end < tokens.len() && tokens[end] == TOK_SEP {
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
            let req_vec =
                featurize_tokens_vq(&ep[req_start..req_end], &vq_tok_emb, dim, vocab_size);
            let mut saci_local_scratch = [0.0_f64; 512];
            let out_fw = forward_pure_weights_with_saci(
                &pure_model,
                &req_vec,
                &ep[req_start..req_end],
                &num_table,
                &vq_tok_emb,
                vocab_size,
                &saci_ckpt,
                &saci_descs,
                &mut saci_local_scratch,
            );

            let mut slot_buf = vec![0.0_f64; 8 * dim];
            let _ = encode_token_slots(
                &ep[req_start..req_end],
                &vq_tok_emb,
                vocab_size,
                dim,
                8,
                &mut slot_buf,
            );
            let mut ncm_feat = Vec::with_capacity(12 * dim);
            ncm_feat.extend_from_slice(&out_fw.ws_state);
            ncm_feat.extend_from_slice(&slot_buf);

            let mut prompt_mat = vec![0.0_f64; NCM_CHUNK_LEN * dim];
            let _ = encode_prompt_matrix(
                &ep[req_start..req_end],
                &vq_tok_emb,
                vocab_size,
                dim,
                NCM_CHUNK_LEN,
                &mut prompt_mat,
            );

            let out_slice = &ep[thought_pos..ep.len().min(thought_pos + NCM_CHUNK_LEN)];
            let mut target_toks = vec![TOK_SEP; NCM_CHUNK_LEN];
            for (pos, &tok) in out_slice.iter().enumerate() {
                target_toks[pos] = tok;
            }
            ws_list.push(ncm_feat);
            prompt_list.push(prompt_mat);
            dom_list.push(out_fw.predicted_domain);
            tok_chunks.push(target_toks);
        }

        let n_eps = tok_chunks.len();
        let mut chunk_list = vec![Vec::new(); n_eps];
        if let Some(ref enc) = vq_enc {
            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(n_eps.max(1));
            let chunk_sz = n_eps.div_ceil(n_threads);
            let tok_ref = &tok_chunks;
            let emb_ref = &vq_tok_emb;
            std::thread::scope(|s| {
                for (t_idx, dst_slice) in chunk_list.chunks_mut(chunk_sz).enumerate() {
                    let base = t_idx * chunk_sz;
                    s.spawn(move || {
                        for (local_i, dst) in dst_slice.iter_mut().enumerate() {
                            *dst = enc.encode_chunk(&tok_ref[base + local_i], emb_ref, vocab_size);
                        }
                    });
                }
            });
        } else {
            for (dst, toks) in chunk_list.iter_mut().zip(tok_chunks.iter()) {
                let mut lat = vec![0.0_f64; NCM_CHUNK_LEN * dim];
                for (pos, &tok) in toks.iter().enumerate() {
                    let tid = tok as usize;
                    if tid < vocab_size {
                        lat[pos * dim..(pos + 1) * dim]
                            .copy_from_slice(&vq_tok_emb[tid * dim..(tid + 1) * dim]);
                    }
                }
                *dst = lat;
            }
        }

        (ws_list, prompt_list, chunk_list, dom_list, tok_chunks)
    };

    let max_ncm_eps = parse_str_flag(args, "--max-ncm-eps")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8192);
    let (ncm_tr_ws, ncm_tr_prompts, ncm_tr_chunks, ncm_tr_doms, _) =
        extract_ncm_pairs(&train_toks, max_ncm_eps);
    drop(train_toks);

    let ncm_in_dim = 12 * dim;
    let ncm_exp_dim = parse_str_flag(args, "--ncm-exp-dim")
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(512);
    let ncm_gen = compile_parallel_ncm_wsp_copy_and_residual(
        &ncm_tr_ws,
        &ncm_tr_prompts,
        &ncm_tr_chunks,
        &ncm_tr_doms,
        saci_descs.len(),
        ncm_in_dim,
        ncm_exp_dim,
        NCM_CHUNK_LEN,
        dim,
        parsed_spec.decoder_lambda,
    )
    .map_err(|e| format!("Parallel NCM WSP compile failed: {e:?}"))?;
    drop(ncm_tr_ws);
    drop(ncm_tr_prompts);
    drop(ncm_tr_chunks);
    drop(ncm_tr_doms);

    let (ncm_va_ws, ncm_va_prompts, ncm_va_chunks, ncm_va_doms, ncm_va_toks) =
        extract_ncm_pairs(&valid_toks, usize::MAX);
    drop(valid_toks);

    // Evaluate parallel NCM chunk cosine similarity & parallel 64-token VQ decoding on held-out valid.bin
    let mut ncm_cos_sum = 0.0_f64;
    let mut ncm_tok_match = 0usize;
    let mut ncm_tok_total = 0usize;
    for idx in 0..ncm_va_ws.len() {
        let z_hat = ncm_gen.generate_concept_chunk_with_prompt(
            &ncm_va_ws[idx],
            &ncm_va_prompts[idx],
            ncm_va_doms[idx],
        );
        let tgt = &ncm_va_chunks[idx];
        let dot: f64 = z_hat.iter().zip(tgt.iter()).map(|(a, b)| a * b).sum();
        let nz: f64 = z_hat.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nt: f64 = tgt.iter().map(|v| v * v).sum::<f64>().sqrt();
        if nz > 1e-12 && nt > 1e-12 {
            ncm_cos_sum += dot / (nz * nt);
        }
        if let Some(ref dec) = vq_dec {
            let decoded_toks = dec.decode_chunk_parallel(&z_hat, NCM_CHUNK_LEN);
            for (p, &g) in decoded_toks.iter().enumerate() {
                if g == ncm_va_toks[idx][p] {
                    ncm_tok_match += 1;
                }
                ncm_tok_total += 1;
            }
        }
    }
    let ncm_mean_cos = ncm_cos_sum / (ncm_va_ws.len().max(1) as f64);
    let ncm_parallel_acc = if ncm_tok_total > 0 {
        100.0 * (ncm_tok_match as f64) / (ncm_tok_total as f64)
    } else {
        0.0
    };

    // 7. Verify Pipeline B exclusivity & iterative PonderNet Workspace halt across validation queries
    let ncm = NcmOutputIr::from_edges(
        vec![
            NcmArtifactKind::Title,
            NcmArtifactKind::MathResults,
            NcmArtifactKind::ValidationSummary,
        ],
        &[(NcmNodeId(0), NcmNodeId(1)), (NcmNodeId(1), NcmNodeId(2))],
    )
    .unwrap();
    let plan = WorkflowPlanIr {
        assignments: &ada_manifest.neuron_assignments,
        saci_ops: &[first_saci_id],
        verify_threshold: 0.01,
    };
    let mut ponder_verified = 0usize;
    for s in &valid_dal {
        let q_masks = compute_exclusive_neuron_masks(
            &ada_model.ir.feature_etf.basis,
            &s.req_vec,
            arch_spec.num_neurons,
            dim,
            0.0,
        );
        verify_exclusivity(&q_masks).map_err(|e| format!("Query ETF exclusivity failed: {e:?}"))?;
        let pb_res = execute_query_pipeline_b(
            &ada_model,
            &plan,
            &ncm,
            &s.req_vec,
            &mut ada_scratch,
            &mut ada_tel,
        )
        .map_err(|e| format!("Pipeline B failed: {e:?}"))?;
        verify_exclusivity(&ada_scratch.wsp.neuron_masks[..arch_spec.num_neurons])
            .map_err(|e| format!("Exclusivity failed: {e:?}"))?;
        if pb_res.workspace_state == WorkspaceState::HaltedVerified {
            ponder_verified += 1;
        }
    }
    assert_eq!(ponder_verified, valid_dal.len());

    // 8. Export ONE unified pure-weights model file (`.safetensors` by default, or `ADAMDL01` `.bin`)
    let saci_arena = saci_ckpt.tensor_slice(TensorId(5)).unwrap();
    let verify_halt = [
        pure_model.verify_bias,
        pure_model.w_halt[0],
        pure_model.w_halt[1],
        pure_model.b_halt,
    ];

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let etf_features_str = (arch_spec.num_neurons * 4).to_string();
    let neurons_str = arch_spec.num_neurons.to_string();
    let dim_str = dim.to_string();
    let layers_str = ada_model.ir.attention.num_layers().to_string();
    let vocab_str = vocab_size.to_string();
    let saci_str = saci_descs.len().to_string();
    let chunk_str = NCM_CHUNK_LEN.to_string();
    let metadata = [
        ("format", "ada-pure-weights"),
        ("model_kind", cfg.model_kind.as_str()),
        ("etf_features", etf_features_str.as_str()),
        ("num_neurons", neurons_str.as_str()),
        ("substrate_dim", dim_str.as_str()),
        ("num_layers", layers_str.as_str()),
        ("vocab_size", vocab_str.as_str()),
        ("saci_circuits", saci_str.as_str()),
        ("ncm_chunk_len", chunk_str.as_str()),
    ];

    let mut named: Vec<NamedTensor<'_>> = vec![
        NamedTensor {
            name: "brain.embed_proj.etf_basis",
            shape: vec![arch_spec.num_neurons * 4, dim],
            data: &ada_model.ir.feature_etf.basis,
        },
        NamedTensor {
            name: "brain.kda.w_knowledge",
            shape: vec![dim, dim],
            data: &pure_model.w_knowledge,
        },
        NamedTensor {
            name: "brain.kda.p_knowledge",
            shape: vec![dim, pure_model.p_knowledge.len() / dim],
            data: &pure_model.p_knowledge,
        },
        NamedTensor {
            name: "brain.mla.w_reasoning",
            shape: vec![2 * dim, dim],
            data: &pure_model.w_reasoning,
        },
        NamedTensor {
            name: "brain.mla.p_reasoning",
            shape: vec![dim, pure_model.p_reasoning.len() / dim],
            data: &pure_model.p_reasoning,
        },
        NamedTensor {
            name: "saci.router.w_route",
            shape: vec![3 * dim, pure_model.w_saci_route.len() / (3 * dim)],
            data: &pure_model.w_saci_route,
        },
        NamedTensor {
            name: "saci.router.p_saci",
            shape: vec![dim, pure_model.p_saci.len() / dim],
            data: &pure_model.p_saci,
        },
        NamedTensor {
            name: "saci.facts.w_facts",
            shape: vec![3 * dim, dim],
            data: &pure_model.w_saci_facts,
        },
        NamedTensor {
            name: "saci.circuits.arena",
            shape: vec![saci_arena.len()],
            data: saci_arena,
        },
        NamedTensor {
            name: "workspace.w_simulation",
            shape: vec![3 * dim, dim],
            data: &pure_model.w_simulation,
        },
        NamedTensor {
            name: "workspace.p_simulation",
            shape: vec![dim, pure_model.p_simulation.len() / dim],
            data: &pure_model.p_simulation,
        },
        NamedTensor {
            name: "bridges.w_gate",
            shape: vec![4, 4],
            data: &pure_model.w_gate,
        },
        NamedTensor {
            name: "bridges.b_gate",
            shape: vec![pure_model.b_gate.len()],
            data: &pure_model.b_gate,
        },
        NamedTensor {
            name: "bridges.verify_halt",
            shape: vec![verify_halt.len()],
            data: &verify_halt,
        },
        NamedTensor {
            name: "ncm.w_coarse",
            shape: vec![ncm_gen.exp_dim, ncm_gen.coarse_len * dim],
            data: &ncm_gen.w_coarse,
        },
        NamedTensor {
            name: "ncm.w_q_attn",
            shape: vec![ncm_gen.num_domains, 4, dim, dim / 4],
            data: &ncm_gen.w_q_attn,
        },
        NamedTensor {
            name: "ncm.w_k_heads",
            shape: vec![4, dim, dim / 4],
            data: &ncm_gen.w_k_heads,
        },
        NamedTensor {
            name: "ncm.rel_pos_bias",
            shape: vec![ncm_gen.num_domains, 4, 127],
            data: &ncm_gen.rel_pos_bias,
        },
        NamedTensor {
            name: "ncm.w_v_attn",
            shape: vec![ncm_gen.num_domains, dim, dim],
            data: &ncm_gen.w_v_attn,
        },
        NamedTensor {
            name: "ncm.w_expand",
            shape: vec![ncm_gen.ws_dim + ncm_gen.num_domains, ncm_gen.exp_dim],
            data: &ncm_gen.w_expand,
        },
        NamedTensor {
            name: "ncm.domain_means",
            shape: vec![ncm_gen.num_domains, NCM_CHUNK_LEN * dim],
            data: &ncm_gen.domain_means,
        },
        NamedTensor {
            name: "ncm.w_ncm_chunk",
            shape: vec![ncm_gen.exp_dim, NCM_CHUNK_LEN * dim],
            data: &ncm_gen.w_ncm_chunk,
        },
        NamedTensor {
            name: "vq.tok_emb.weight",
            shape: vec![vocab_size, dim],
            data: &vq_tok_emb,
        },
        NamedTensor {
            name: "vq.head_bias",
            shape: vec![vocab_size],
            data: &vq_head_bias,
        },
    ];

    if let Some(ref dec) = vq_dec {
        named.extend([
            NamedTensor {
                name: "vq.ln_dec.gamma",
                shape: vec![dim],
                data: &dec.ln_dec_gamma,
            },
            NamedTensor {
                name: "vq.ln_dec.beta",
                shape: vec![dim],
                data: &dec.ln_dec_beta,
            },
            NamedTensor {
                name: "vq.dec1.ln1.gamma",
                shape: vec![dim],
                data: &dec.dec1.ln1_gamma,
            },
            NamedTensor {
                name: "vq.dec1.ln1.beta",
                shape: vec![dim],
                data: &dec.dec1.ln1_beta,
            },
            NamedTensor {
                name: "vq.dec1.ln2.gamma",
                shape: vec![dim],
                data: &dec.dec1.ln2_gamma,
            },
            NamedTensor {
                name: "vq.dec1.ln2.beta",
                shape: vec![dim],
                data: &dec.dec1.ln2_beta,
            },
            NamedTensor {
                name: "vq.dec2.ln1.gamma",
                shape: vec![dim],
                data: &dec.dec2.ln1_gamma,
            },
            NamedTensor {
                name: "vq.dec2.ln1.beta",
                shape: vec![dim],
                data: &dec.dec2.ln1_beta,
            },
            NamedTensor {
                name: "vq.dec2.ln2.gamma",
                shape: vec![dim],
                data: &dec.dec2.ln2_gamma,
            },
            NamedTensor {
                name: "vq.dec2.ln2.beta",
                shape: vec![dim],
                data: &dec.dec2.ln2_beta,
            },
            NamedTensor {
                name: "vq.dec1.wq.weight",
                shape: vec![dim, dim],
                data: &dec.dec1.wq,
            },
            NamedTensor {
                name: "vq.dec1.wq.bias",
                shape: vec![dim],
                data: &dec.dec1.bq,
            },
            NamedTensor {
                name: "vq.dec1.wk.weight",
                shape: vec![dim, dim],
                data: &dec.dec1.wk,
            },
            NamedTensor {
                name: "vq.dec1.wk.bias",
                shape: vec![dim],
                data: &dec.dec1.bk,
            },
            NamedTensor {
                name: "vq.dec1.wv.weight",
                shape: vec![dim, dim],
                data: &dec.dec1.wv,
            },
            NamedTensor {
                name: "vq.dec1.wv.bias",
                shape: vec![dim],
                data: &dec.dec1.bv,
            },
            NamedTensor {
                name: "vq.dec1.wo.weight",
                shape: vec![dim, dim],
                data: &dec.dec1.wo,
            },
            NamedTensor {
                name: "vq.dec1.wo.bias",
                shape: vec![dim],
                data: &dec.dec1.bo,
            },
            NamedTensor {
                name: "vq.dec1.ff1.weight",
                shape: vec![dim, dec.dec1.ff_dim],
                data: &dec.dec1.ff1_w,
            },
            NamedTensor {
                name: "vq.dec1.ff1.bias",
                shape: vec![dec.dec1.ff_dim],
                data: &dec.dec1.ff1_b,
            },
            NamedTensor {
                name: "vq.dec1.ff2.weight",
                shape: vec![dec.dec1.ff_dim, dim],
                data: &dec.dec1.ff2_w,
            },
            NamedTensor {
                name: "vq.dec1.ff2.bias",
                shape: vec![dim],
                data: &dec.dec1.ff2_b,
            },
            NamedTensor {
                name: "vq.dec2.wq.weight",
                shape: vec![dim, dim],
                data: &dec.dec2.wq,
            },
            NamedTensor {
                name: "vq.dec2.wq.bias",
                shape: vec![dim],
                data: &dec.dec2.bq,
            },
            NamedTensor {
                name: "vq.dec2.wk.weight",
                shape: vec![dim, dim],
                data: &dec.dec2.wk,
            },
            NamedTensor {
                name: "vq.dec2.wk.bias",
                shape: vec![dim],
                data: &dec.dec2.bk,
            },
            NamedTensor {
                name: "vq.dec2.wv.weight",
                shape: vec![dim, dim],
                data: &dec.dec2.wv,
            },
            NamedTensor {
                name: "vq.dec2.wv.bias",
                shape: vec![dim],
                data: &dec.dec2.bv,
            },
            NamedTensor {
                name: "vq.dec2.wo.weight",
                shape: vec![dim, dim],
                data: &dec.dec2.wo,
            },
            NamedTensor {
                name: "vq.dec2.wo.bias",
                shape: vec![dim],
                data: &dec.dec2.bo,
            },
            NamedTensor {
                name: "vq.dec2.ff1.weight",
                shape: vec![dim, dec.dec2.ff_dim],
                data: &dec.dec2.ff1_w,
            },
            NamedTensor {
                name: "vq.dec2.ff1.bias",
                shape: vec![dec.dec2.ff_dim],
                data: &dec.dec2.ff1_b,
            },
            NamedTensor {
                name: "vq.dec2.ff2.weight",
                shape: vec![dec.dec2.ff_dim, dim],
                data: &dec.dec2.ff2_w,
            },
            NamedTensor {
                name: "vq.dec2.ff2.bias",
                shape: vec![dim],
                data: &dec.dec2.ff2_b,
            },
        ]);
    }

    let total_weights: usize = named.iter().map(|t| t.data.len()).sum();
    if matches!(cfg.dtype, WeightDtype::BF16 | WeightDtype::F16) {
        let typed: Vec<TypedNamedTensor<'_>> = named
            .iter()
            .map(|t| {
                let keep_f32 = t.name == "saci.circuits.arena"
                    || t.name.starts_with("bridges.")
                    || t.name.contains(".p_")
                    || t.name.ends_with(".bias")
                    || t.name.ends_with("_bias")
                    || t.name.ends_with(".gamma")
                    || t.name.ends_with(".beta");
                TypedNamedTensor {
                    name: t.name,
                    shape: t.shape.clone(),
                    dtype: if keep_f32 {
                        WeightDtype::F32
                    } else {
                        cfg.dtype
                    },
                    data: t.data,
                }
            })
            .collect();
        let (_, health) = write_mixed_safetensors(&out_path, &metadata, &typed)?;
        println!(
            "Health Gate: passed={} nonfinite={} f16_overflow={} max_abs={:.4} outliers={}",
            health.passed,
            health.nonfinite_count,
            health.f16_overflow_count,
            health.max_abs_weight,
            health.outlier_tensor_count
        );
    } else {
        write_safetensors(&out_path, cfg.dtype, &metadata, &named)?;
    }

    println!(
        "Built {} ({} bytes, {} pure weights [{}], model={}, etf_features={}, {} SACI circuits from {}, valid_acc={}/{} [{:.2}%], all_4_verified={}/{}, ponder_verified={}/{}, ncm_mean_cos={:.4} ncm_parallel_tok_acc={:.2}% ({}/{} tokens), elapsed={} ms)",
        out_path.display(),
        fs::metadata(&out_path)?.len(),
        total_weights,
        cfg.dtype.as_str(),
        cfg.model_kind.as_str(),
        arch_spec.num_neurons * 4,
        saci_descs.len(),
        saci_dir.display(),
        pw_correct,
        valid_dal.len(),
        100.0 * (pw_correct as f64) / (valid_dal.len() as f64),
        pw_verified,
        valid_dal.len(),
        ponder_verified,
        valid_dal.len(),
        ncm_mean_cos,
        ncm_parallel_acc,
        ncm_tok_match,
        ncm_tok_total,
        t0.elapsed().as_millis()
    );

    Ok(())
}
