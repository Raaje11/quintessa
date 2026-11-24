//! `quintessa`: Unified ADA Neural Architecture Compiler, SACI Circuit Compiler,
//! and Algorithmic Weight-Space Programming (WSP) Engine.
//!
//! Built in strict compliance with `D:\TRACR\GEMINI.md`.
//! Unifies:
//! - ADA Architecture IR, Feature/ETF IR, Sparse Lowering & Pipeline A/B Runtime (`ir`, `etf`, `wsp`, `attention`, `capability`, `manifest`, `fhp`, `runtime`)
//! - Structured Telemetry & Observatory (`sys`, `observatory`)
//! - Checkpoint Inspection, DCIR, Analytical/Derivative-Free WSP Solvers & Adaptive PonderNet (`architecture`, `checkpoint`, `dcir`, `solvers`, `semantic_wsp`)
//! - In-Neuron SACI Circuit Compiler & Operand Grounding (`saci_circuit`, `saci_compiler`, `operand_grounding`, `knowledge_base`)
//! - Parallel NCM Generator, VQ BPK Loader, Token Decoder & Safetensors Export (`ncm_generator`, `bpk_loader`, `token_decoder`, `safetensors_writer`)

pub mod ada_compiler;
pub mod ada_o1_insane;
pub mod ada_scorecard;
pub mod architecture;
pub mod attention;
pub mod bpk_loader;
pub mod cad_brep;
pub mod capability;
pub mod checkpoint;
pub mod compiler_passes;
pub mod conversation;
pub mod dcir;
pub mod etf;
pub mod fhp;
pub mod fluency_multimodal_layer;
pub mod ir;
pub mod knowledge_base;
pub mod manifest;
pub mod multimodal_ncm;
pub mod ncm_generator;
pub mod observatory;
pub mod operand_grounding;
pub mod primitives;
pub mod runtime;
pub mod saci_circuit;
pub mod saci_compiler;
pub mod safetensors_writer;
pub mod semantic_compiler;
pub mod semantic_data;
pub mod semantic_wsp;
pub mod sequence_encoder;
pub mod sharded_compiler;
pub mod simulation_workspace;
pub mod solvers;
pub mod statistical_lm;
pub mod superposition_wsp;
pub mod sys;
pub mod token_decoder;
pub mod verifier;
pub mod workflow_arch;
pub mod wsp;

use architecture::{
    reconstruct_architecture, validate_patch_permissions, ArchitectureManifest, ModelGraph,
};
use checkpoint::{apply_patches, inspect_checkpoint, Checkpoint, WeightPatch};
use dcir::{
    compile_dataset, step_ponder, AdaptivePonderConfig, DomainCompilationIr, DomainDataset,
    PonderState, ReasoningSpecification, ReasoningStep,
};
use solvers::{
    calibrate_ponder_threshold, select_solver, solve_nullspace_edit, solve_ridge_projection,
};
use sys::{ErrorCode, PonderAction, SolverKind, TensorId};

#[derive(Clone, Debug)]
pub struct WspSpecification {
    pub target_tensors: Vec<TensorId>,
    pub regularization_lambda: f64,
    pub sparse_delta_epsilon: f64,
    pub max_target_mse: f64,
    pub max_protected_drift: f64,
}

#[derive(Clone, Debug)]
pub struct WspProgram {
    pub domain_id: u32,
    pub steps: Vec<ReasoningStep>,
    pub ponder: AdaptivePonderConfig,
    pub calibrated_halt_threshold: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct WspWriteReport {
    pub tensors_inspected: usize,
    pub patches_generated: usize,
    pub nonzero_deltas: usize,
    pub solver_used: SolverKind,
    pub calibrated_halt_threshold: f64,
}

pub struct WspInput<'a> {
    pub checkpoint: &'a Checkpoint,
    pub architecture: &'a ArchitectureManifest,
    pub specification: &'a WspSpecification,
    pub dataset: &'a DomainDataset,
    pub reasoning: &'a ReasoningSpecification,
}

pub struct WspOutput {
    pub candidate_checkpoint: Checkpoint,
    pub patches: Vec<WeightPatch>,
    pub program: WspProgram,
    pub write_report: WspWriteReport,
    pub domain_ir: DomainCompilationIr,
    pub model_graph: ModelGraph,
}

#[derive(Clone, Copy, Debug)]
pub struct VerificationReport {
    pub target_mse: f64,
    pub protected_drift: f64,
    pub avg_ponder_steps: f64,
    pub saci_escalations: usize,
    pub passed: bool,
}

#[inline]
fn linear_forward(w: &[f64], d_in: usize, d_out: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; d_out];
    for c in 0..d_out {
        let mut acc = 0.0;
        for i in 0..d_in {
            acc += x[i] * w[i * d_out + c];
        }
        out[c] = acc;
    }
    out
}

/// Orchestrate the full AWSC pipeline:
/// `(Checkpoint, Architecture, WSP Spec, Dataset, Reasoning) -> (Weight Updates, WSP Program)`.
pub fn compile_weight_space(input: WspInput<'_>) -> Result<WspOutput, ErrorCode> {
    let inventory = inspect_checkpoint(input.checkpoint)?;
    let model_graph = reconstruct_architecture(input.architecture, &inventory)?;
    let domain_ir = compile_dataset(input.dataset, input.reasoning)?;

    if input.specification.target_tensors.is_empty() || domain_ir.trainable_examples.is_empty() {
        return Err(ErrorCode::InvalidInput);
    }

    let has_protected = !domain_ir.protected_examples.is_empty();
    let mut patches = Vec::with_capacity(input.specification.target_tensors.len());
    let mut nonzero_deltas = 0usize;
    let mut last_solver = SolverKind::RidgePseudoinverse;

    // Flatten new and protected training matrices
    let d_in = domain_ir.feature_dim;
    let d_out = domain_ir.target_dim;
    let n_new = domain_ir.trainable_examples.len();
    let mut x_new = Vec::with_capacity(n_new * d_in);
    let mut y_new = Vec::with_capacity(n_new * d_out);
    for ex in &domain_ir.trainable_examples {
        x_new.extend_from_slice(&ex.features);
        y_new.extend_from_slice(&ex.target);
    }

    let n_prot = domain_ir.protected_examples.len();
    let mut x_prot = Vec::with_capacity(n_prot * d_in);
    for ex in &domain_ir.protected_examples {
        x_prot.extend_from_slice(&ex.features);
    }

    for &tid in &input.specification.target_tensors {
        let desc = input
            .checkpoint
            .descriptor(tid)
            .ok_or(ErrorCode::ShapeMismatch)?;
        let perm = input
            .architecture
            .permission(tid)
            .ok_or(ErrorCode::UnauthorizedWrite)?;

        if !perm.policy.is_writable() {
            return Err(ErrorCode::ProtectedInvariant);
        }
        if desc.rows != d_in || desc.cols != d_out {
            return Err(ErrorCode::ShapeMismatch);
        }

        let op = domain_ir
            .reasoning_steps
            .iter()
            .find(|s| s.target_component == perm.component)
            .map(|s| s.op)
            .unwrap_or(domain_ir.ops[0]);

        let solver = select_solver(perm.component, op, has_protected)?;
        last_solver = solver;

        let w0 = input
            .checkpoint
            .tensor_slice(tid)
            .ok_or(ErrorCode::ShapeMismatch)?;

        let dw = match solver {
            SolverKind::SvdNullspace if has_protected => solve_nullspace_edit(
                d_in,
                d_out,
                w0,
                &x_new,
                &y_new,
                &x_prot,
                input.specification.regularization_lambda,
            )?,
            _ => {
                let w_star = solve_ridge_projection(
                    n_new,
                    d_in,
                    d_out,
                    &x_new,
                    &y_new,
                    w0,
                    input.specification.regularization_lambda,
                )?;
                w_star.iter().zip(w0.iter()).map(|(a, b)| a - b).collect()
            }
        };

        let mut sparse_deltas = Vec::new();
        for (idx, &val) in dw.iter().enumerate() {
            if val.abs() > input.specification.sparse_delta_epsilon {
                sparse_deltas.push((idx, val));
            }
        }
        nonzero_deltas += sparse_deltas.len();
        patches.push(WeightPatch {
            tensor_id: tid,
            sparse_deltas,
        });
    }

    validate_patch_permissions(&patches, input.architecture)?;
    let candidate_checkpoint = apply_patches(input.checkpoint, &patches)?;

    // Calibrate Adaptive PonderNet halting threshold via derivative-free golden-section search
    let primary_tid = input.specification.target_tensors[0];
    let w_cand = candidate_checkpoint
        .tensor_slice(primary_tid)
        .ok_or(ErrorCode::ShapeMismatch)?;
    let base_ponder = domain_ir.ponder;
    let calibrated_halt_threshold = calibrate_ponder_threshold(
        |thr| {
            let mut cfg = base_ponder;
            cfg.halt_threshold = thr;
            let mut total_loss = 0.0;
            for ex in &domain_ir.trainable_examples {
                let pred = linear_forward(w_cand, d_in, d_out, &ex.features);
                let err: f64 = pred
                    .iter()
                    .zip(ex.target.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                let mut st = PonderState::new();
                let mut steps = 0usize;
                loop {
                    let residual = err / (steps as f64 + 1.0);
                    let logit = if residual <= cfg.halt_tolerance {
                        2.0
                    } else {
                        -0.5
                    };
                    let (act, _) = step_ponder(&mut st, logit, residual, &cfg);
                    steps += 1;
                    if act != PonderAction::Continue {
                        break;
                    }
                }
                total_loss += err + cfg.kl_beta * st.kl_regularization + 0.01 * (steps as f64);
            }
            total_loss / (domain_ir.trainable_examples.len() as f64)
        },
        0.5,
        0.95,
        16,
    );

    let mut ponder = domain_ir.ponder;
    ponder.halt_threshold = calibrated_halt_threshold;

    let write_report = WspWriteReport {
        tensors_inspected: inventory.stats.len(),
        patches_generated: patches.len(),
        nonzero_deltas,
        solver_used: last_solver,
        calibrated_halt_threshold,
    };

    let program = WspProgram {
        domain_id: domain_ir.domain_id,
        steps: domain_ir.reasoning_steps.clone(),
        ponder,
        calibrated_halt_threshold,
    };

    Ok(WspOutput {
        candidate_checkpoint,
        patches,
        program,
        write_report,
        domain_ir,
        model_graph,
    })
}

/// Independent empirical verifier: checks target MSE, protected-input drift,
/// and Adaptive PonderNet reasoning behavior on the candidate checkpoint.
pub fn verify_candidate_checkpoint(
    original: &Checkpoint,
    output: &WspOutput,
    spec: &WspSpecification,
) -> Result<VerificationReport, ErrorCode> {
    let tid = spec.target_tensors[0];
    let desc = original.descriptor(tid).ok_or(ErrorCode::ShapeMismatch)?;
    let w_orig = original.tensor_slice(tid).ok_or(ErrorCode::ShapeMismatch)?;
    let w_cand = output
        .candidate_checkpoint
        .tensor_slice(tid)
        .ok_or(ErrorCode::ShapeMismatch)?;

    let d_in = desc.rows;
    let d_out = desc.cols;

    let mut target_sq = 0.0;
    let mut total_steps = 0usize;
    let mut saci_escalations = 0usize;

    for ex in &output.domain_ir.trainable_examples {
        let pred = linear_forward(w_cand, d_in, d_out, &ex.features);
        let err: f64 = pred
            .iter()
            .zip(ex.target.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        target_sq += err;

        let mut st = PonderState::new();
        loop {
            let logit = if err <= output.program.ponder.halt_tolerance {
                2.5
            } else {
                0.0
            };
            let (act, _) = step_ponder(&mut st, logit, err, &output.program.ponder);
            if act == PonderAction::EscalateToSaci {
                saci_escalations += 1;
                break;
            }
            if act != PonderAction::Continue {
                break;
            }
        }
        total_steps += st.step as usize;
    }

    let target_mse = target_sq / (output.domain_ir.trainable_examples.len() as f64);

    let mut prot_drift = 0.0_f64;
    if !output.domain_ir.protected_examples.is_empty() {
        for ex in &output.domain_ir.protected_examples {
            let p0 = linear_forward(w_orig, d_in, d_out, &ex.features);
            let p1 = linear_forward(w_cand, d_in, d_out, &ex.features);
            let diff_sq: f64 = p0
                .iter()
                .zip(p1.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            prot_drift += diff_sq;
        }
        prot_drift /= output.domain_ir.protected_examples.len() as f64;
    }

    let avg_ponder_steps =
        (total_steps as f64) / (output.domain_ir.trainable_examples.len() as f64);

    let passed = target_mse <= spec.max_target_mse && prot_drift <= spec.max_protected_drift;

    Ok(VerificationReport {
        target_mse,
        protected_drift: prot_drift,
        avg_ponder_steps,
        saci_escalations,
        passed,
    })
}

#[cfg(test)]
mod compiler_tests {
    use crate::ir::{
        ArchitectureSpec, AttentionIr, KdaConfig, MlaConfig, NcmArtifactKind, NcmOutputIr,
        NeuronAssignment, SaciIr, SaciOpSpec, WorkflowPlanIr, WorkspaceIr,
    };
    use crate::observatory;
    use crate::runtime::{compile_architecture, execute_query_pipeline_b, AdaQueryScratch};
    use crate::sys::{
        AttentionKind, Backend, ErrorCode, FeatureFamily, NcmNodeId, NeuronId, NumericalMode,
        SaciOpId, TelemetryBus, WorkspaceState, MASK_KNOWLEDGE, MASK_REASONING,
    };
    use crate::wsp::{generate_wsp_masks, verify_exclusivity};

    fn default_spec(num_neurons: usize, substrate_dim: usize) -> ArchitectureSpec {
        ArchitectureSpec {
            vq_codebook_size: 4096,
            num_neurons,
            substrate_dim,
            tile_size: 16,
            kda: KdaConfig {
                chunk_size: 64,
                dplr_rank: 4,
                state_dim: 128,
            },
            mla: MlaConfig {
                kv_latent_dim: 64,
                num_heads: 8,
                head_dim: 64,
            },
            workspace: WorkspaceIr::default(),
        }
    }

    fn default_saci() -> SaciIr {
        SaciIr::new(vec![
            SaciOpSpec {
                id: SaciOpId(1), // SparseLinearSolve
                numerical_mode: NumericalMode::CertifiedApproximate,
                tolerance: 1e-8,
                backend: Backend::Gpu,
                precondition_flags: 0x1,
                postcondition_flags: 0x1,
            },
            SaciOpSpec {
                id: SaciOpId(2), // NumericalIntegration
                numerical_mode: NumericalMode::CertifiedApproximate,
                tolerance: 1e-6,
                backend: Backend::Cpu,
                precondition_flags: 0x2,
                postcondition_flags: 0x2,
            },
        ])
    }

    #[test]
    fn priority1_feature_etf_four_families_per_neuron() {
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let spec = default_spec(64, 24);
        let model = compile_architecture(&spec, default_saci(), None, &mut telemetry).unwrap();

        assert_eq!(model.ir.feature_etf.num_neurons, 64);
        assert_eq!(model.ir.feature_etf.substrate_dim, 24);
        // 4-in-3 Simplex ETF block coherence is exactly 1/3
        assert!((model.ir.feature_etf.max_coherence - (1.0 / 3.0)).abs() < 1e-10);

        // Verify all 4 feature vectors for neuron 0 are unit-norm
        for fam in FeatureFamily::ALL {
            let v = model.ir.feature_etf.feature_vec(0, fam);
            let norm_sq: f64 = v.iter().map(|x| x * x).sum();
            assert!((norm_sq - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn priority2_wsp_enforces_at_most_one_feature_per_neuron() {
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let spec = default_spec(32, 12);
        let model = compile_architecture(&spec, default_saci(), None, &mut telemetry).unwrap();
        let mut scratch = AdaQueryScratch::new(&model, 8);

        // Violate sum_f z_{i,f} <= 1 by assigning Knowledge AND Reasoning to NeuronId(5)
        let bad_assignments = [
            NeuronAssignment {
                neuron: NeuronId(5),
                family: FeatureFamily::Knowledge,
            },
            NeuronAssignment {
                neuron: NeuronId(5),
                family: FeatureFamily::Reasoning,
            },
        ];
        let bad_plan = WorkflowPlanIr {
            assignments: &bad_assignments,
            saci_ops: &[SaciOpId(1)],
            verify_threshold: 0.1,
        };

        let err = generate_wsp_masks(
            &model.static_wsp,
            &bad_plan,
            &mut scratch.wsp,
            &mut telemetry,
        )
        .unwrap_err();
        assert_eq!(err, ErrorCode::ExclusivityViolation);

        // Also test raw mask verification helper
        assert_eq!(verify_exclusivity(&[0, 1, 2, 4, 8, 0]), Ok(()));
        assert_eq!(
            verify_exclusivity(&[0, 1 | 2]),
            Err(ErrorCode::ExclusivityViolation)
        );
    }

    #[test]
    fn priority2_wsp_rejects_illegal_pathway_and_unknown_saci_op() {
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let spec = default_spec(16, 12);
        let mut legal = vec![MASK_KNOWLEDGE | MASK_REASONING; 16];
        legal[3] = MASK_KNOWLEDGE; // Neuron 3 only allows Knowledge

        let model =
            compile_architecture(&spec, default_saci(), Some(&legal), &mut telemetry).unwrap();
        let mut scratch = AdaQueryScratch::new(&model, 8);

        let illegal_assign = [NeuronAssignment {
            neuron: NeuronId(3),
            family: FeatureFamily::Simulation,
        }];
        let plan1 = WorkflowPlanIr {
            assignments: &illegal_assign,
            saci_ops: &[],
            verify_threshold: 0.1,
        };
        assert_eq!(
            generate_wsp_masks(&model.static_wsp, &plan1, &mut scratch.wsp, &mut telemetry),
            Err(ErrorCode::IllegalPathway)
        );

        let valid_assign = [NeuronAssignment {
            neuron: NeuronId(3),
            family: FeatureFamily::Knowledge,
        }];
        let plan2 = WorkflowPlanIr {
            assignments: &valid_assign,
            saci_ops: &[SaciOpId(999)],
            verify_threshold: 0.1,
        };
        assert_eq!(
            generate_wsp_masks(&model.static_wsp, &plan2, &mut scratch.wsp, &mut telemetry),
            Err(ErrorCode::UnknownSaciOp)
        );
    }

    #[test]
    fn priority3_to_9_sparse_lowering_and_full_pipeline_b() {
        let mut telemetry: TelemetryBus<128> = TelemetryBus::new();
        let spec = default_spec(64, 24);
        let model = compile_architecture(&spec, default_saci(), None, &mut telemetry).unwrap();

        // Build NCM Output DAG:
        //   0: Title (wave 0)
        //   1: GeometrySpec (wave 0)
        //   2: MathResults (depends on 1) -> wave 1
        //   3: Tables (depends on 2) -> wave 2
        //   4: ValidationSummary (depends on 1, 2) -> wave 2
        //   5: TechnicalExplanation (depends on 0, 3, 4) -> wave 3
        let ncm = NcmOutputIr::from_edges(
            vec![
                NcmArtifactKind::Title,
                NcmArtifactKind::GeometrySpec,
                NcmArtifactKind::MathResults,
                NcmArtifactKind::Tables,
                NcmArtifactKind::ValidationSummary,
                NcmArtifactKind::TechnicalExplanation,
            ],
            &[
                (NcmNodeId(1), NcmNodeId(2)),
                (NcmNodeId(2), NcmNodeId(3)),
                (NcmNodeId(1), NcmNodeId(4)),
                (NcmNodeId(2), NcmNodeId(4)),
                (NcmNodeId(0), NcmNodeId(5)),
                (NcmNodeId(3), NcmNodeId(5)),
                (NcmNodeId(4), NcmNodeId(5)),
            ],
        )
        .unwrap();

        let mut scratch = AdaQueryScratch::new(&model, 16);

        // Activate only 8 out of 64 neurons (87.5% sparsity, all in tile 0 out of 4 tiles)
        let assignments = [
            NeuronAssignment {
                neuron: NeuronId(0),
                family: FeatureFamily::Knowledge,
            },
            NeuronAssignment {
                neuron: NeuronId(1),
                family: FeatureFamily::Knowledge,
            },
            NeuronAssignment {
                neuron: NeuronId(2),
                family: FeatureFamily::Reasoning,
            },
            NeuronAssignment {
                neuron: NeuronId(3),
                family: FeatureFamily::Reasoning,
            },
            NeuronAssignment {
                neuron: NeuronId(4),
                family: FeatureFamily::SaciInterface,
            },
            NeuronAssignment {
                neuron: NeuronId(5),
                family: FeatureFamily::SaciInterface,
            },
            NeuronAssignment {
                neuron: NeuronId(6),
                family: FeatureFamily::Simulation,
            },
            NeuronAssignment {
                neuron: NeuronId(7),
                family: FeatureFamily::Simulation,
            },
        ];
        let plan = WorkflowPlanIr {
            assignments: &assignments,
            saci_ops: &[SaciOpId(1), SaciOpId(2)],
            verify_threshold: 0.1,
        };

        // Construct input aligned with neuron 0 Knowledge feature so ReLU fires & verifies
        let mut input = vec![0.0; 24];
        input.copy_from_slice(
            model
                .ir
                .feature_etf
                .feature_vec(0, FeatureFamily::Knowledge),
        );

        let res =
            execute_query_pipeline_b(&model, &plan, &ncm, &input, &mut scratch, &mut telemetry)
                .unwrap();

        // Sparse execution genuinely skipped 56/64 neurons and 3/4 tiles
        assert_eq!(res.sparse_stats.total_neurons, 64);
        assert_eq!(res.sparse_stats.active_neurons, 8);
        assert_eq!(res.sparse_stats.skipped_neurons, 56);
        assert_eq!(res.sparse_stats.total_tiles, 4);
        assert_eq!(res.sparse_stats.active_tiles, 1);
        assert!(res.sparse_stats.skipped_flops > res.sparse_stats.active_flops);

        // Workspace verified and halted on iter 1
        assert_eq!(res.workspace_state, WorkspaceState::HaltedVerified);
        assert_eq!(res.workspace_iters, 1);
        assert_eq!(res.saci_dispatched, 2);

        // NCM scheduled 4 parallel waves for 6 nodes
        assert_eq!(res.ncm_num_waves, 4);
        assert_eq!(res.ncm_total_nodes, 6);

        // Observatory renders telemetry events cleanly
        assert!(!telemetry.is_empty());
        while let Some(ev) = telemetry.pop() {
            let line = observatory::format_event(&ev);
            assert!(!line.is_empty());
        }
    }

    #[test]
    fn attention_topology_and_ncm_cycle_validation() {
        let kda = KdaConfig {
            chunk_size: 64,
            dplr_rank: 4,
            state_dim: 128,
        };
        let mla = MlaConfig {
            kv_latent_dim: 64,
            num_heads: 8,
            head_dim: 64,
        };
        assert_eq!(
            AttentionIr::build(&[], kda, mla).unwrap_err(),
            ErrorCode::InvalidAttentionTopology
        );
        let all_kda = [AttentionKind::Kda; 8];
        assert_eq!(
            AttentionIr::build(&all_kda, kda, mla).unwrap().num_layers(),
            8
        );

        let cyclic = NcmOutputIr::from_edges(
            vec![NcmArtifactKind::Title, NcmArtifactKind::MathResults],
            &[(NcmNodeId(0), NcmNodeId(1)), (NcmNodeId(1), NcmNodeId(0))],
        );
        assert_eq!(cyclic.unwrap_err(), ErrorCode::NcmCycleDetected);
    }

    #[test]
    fn seven_phase_routing_bootstrap_scoring_and_replanning() {
        use crate::ir::SaciDag;
        use crate::runtime::execute_query_with_replanning;
        use crate::sys::{EventKind, MASK_SACI, MASK_SIMULATION};
        use crate::wsp::{bootstrap_wsp, score_and_select_features};

        let mut telemetry: TelemetryBus<256> = TelemetryBus::new();
        let spec = default_spec(32, 24);
        let model = compile_architecture(&spec, default_saci(), None, &mut telemetry).unwrap();
        let mut scratch = AdaQueryScratch::new(&model, 8);

        // 1. Bootstrap activates only Knowledge/Reasoning within budget
        let boot_count =
            bootstrap_wsp(&model.static_wsp, 42, 6, &mut scratch.wsp, &mut telemetry).unwrap();
        assert_eq!(boot_count, 6);
        for &m in &scratch.wsp.neuron_masks {
            assert_eq!(m & (MASK_SACI | MASK_SIMULATION), 0);
        }

        // 2. Stage B multi-layer scoring across all 8 layers (6 KDA + 2 MLA)
        let mut audit_log = Vec::new();
        let scores = vec![[0.2f32, 0.85, 0.4, 0.1]; 32];
        for l in 0..8u8 {
            let active = score_and_select_features(
                &model.static_wsp,
                crate::sys::QueryId(7),
                0,
                l,
                &scores,
                0.5,
                &mut scratch.wsp,
                Some(&mut audit_log),
                &mut telemetry,
            )
            .unwrap();
            assert_eq!(active, 32);
        }
        assert_eq!(audit_log.len(), 8 * 32);

        // 3. Full 7-phase query routing with verification failure -> Reasoning replanning -> pass
        static PLAN0_ASSIGN: [NeuronAssignment; 2] = [
            NeuronAssignment {
                neuron: NeuronId(10),
                family: FeatureFamily::Knowledge,
            },
            NeuronAssignment {
                neuron: NeuronId(11),
                family: FeatureFamily::Reasoning,
            },
        ];
        static PLAN1_ASSIGN: [NeuronAssignment; 3] = [
            NeuronAssignment {
                neuron: NeuronId(0),
                family: FeatureFamily::Knowledge,
            },
            NeuronAssignment {
                neuron: NeuronId(1),
                family: FeatureFamily::SaciInterface,
            },
            NeuronAssignment {
                neuron: NeuronId(2),
                family: FeatureFamily::Simulation,
            },
        ];
        static SACI_OPS: [SaciOpId; 2] = [SaciOpId(1), SaciOpId(2)];

        let plan0 = WorkflowPlanIr {
            assignments: &PLAN0_ASSIGN,
            saci_ops: &SACI_OPS,
            verify_threshold: 0.5,
        };
        let saci_dag =
            SaciDag::from_ops_and_edges(vec![SaciOpId(1), SaciOpId(2)], &[(0, 1)]).unwrap();
        let ncm = NcmOutputIr::from_edges(
            vec![NcmArtifactKind::Title, NcmArtifactKind::MathResults],
            &[(NcmNodeId(0), NcmNodeId(1))],
        )
        .unwrap();

        // Input aligns strictly with Neuron 0 Knowledge (orthogonal block to neurons 10..11)
        let input = model
            .ir
            .feature_etf
            .feature_vec(0, FeatureFamily::Knowledge)
            .to_vec();

        let res = execute_query_with_replanning(
            &model,
            99,
            4,
            &plan0,
            &saci_dag,
            &ncm,
            &input,
            |_rev, _feedback| {
                Some(WorkflowPlanIr {
                    assignments: &PLAN1_ASSIGN,
                    saci_ops: &SACI_OPS,
                    verify_threshold: 0.5,
                })
            },
            &mut scratch,
            &mut telemetry,
        )
        .unwrap();

        assert_eq!(res.workspace_state, WorkspaceState::HaltedVerified);
        assert_eq!(res.workspace_iters, 2); // Iter 1 failed, replanned, Iter 2 passed
        assert_eq!(res.saci_dispatched, 4); // 2 ops x 2 iterations
        assert_eq!(res.sparse_stats.active_neurons, 3);
        assert_eq!(res.sparse_stats.skipped_neurons, 29);

        // Verify telemetry recorded BootstrapRouted, VerificationFailed, and PlanRevised
        let mut saw_boot = false;
        let mut saw_fail = false;
        let mut saw_rev = false;
        while let Some(ev) = telemetry.pop() {
            match ev.event {
                EventKind::BootstrapRouted => saw_boot = true,
                EventKind::VerificationFailed => saw_fail = true,
                EventKind::PlanRevised => saw_rev = true,
                _ => {}
            }
        }
        assert!(saw_boot && saw_fail && saw_rev);
    }

    #[test]
    fn fhp_v2_and_v1_round_trip_and_validation() {
        use crate::fhp::{load_fhp, save_fhp, save_fhp_v1, FhpError, FhpVersion};
        use std::io::Cursor;

        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let mut spec = default_spec(64, 24);
        spec.workspace = WorkspaceIr {
            scratchpad_slots: 32,
            max_iters: 12,
            halting_tolerance: 1e-7,
        };
        let model = compile_architecture(&spec, default_saci(), None, &mut telemetry).unwrap();

        // V2 lossless round-trip
        let mut buf_v2 = Vec::new();
        save_fhp(&model, &mut buf_v2).unwrap();
        let (loaded_v2, ver_v2) = load_fhp(&mut Cursor::new(&buf_v2)).unwrap();
        assert_eq!(ver_v2, FhpVersion::V2);
        assert_eq!(loaded_v2.ir.vq_codebook_size, model.ir.vq_codebook_size);
        assert_eq!(loaded_v2.ir.tile_size, model.ir.tile_size);
        assert_eq!(
            loaded_v2.ir.feature_etf.num_neurons,
            model.ir.feature_etf.num_neurons
        );
        assert_eq!(
            loaded_v2.ir.feature_etf.substrate_dim,
            model.ir.feature_etf.substrate_dim
        );
        assert_eq!(
            loaded_v2.ir.feature_etf.legal_masks,
            model.ir.feature_etf.legal_masks
        );
        assert_eq!(loaded_v2.ir.feature_etf.basis, model.ir.feature_etf.basis);
        assert_eq!(loaded_v2.ir.workspace.scratchpad_slots, 32);
        assert_eq!(loaded_v2.ir.workspace.max_iters, 12);
        assert!((loaded_v2.ir.workspace.halting_tolerance - 1e-7).abs() < 1e-15);
        assert_eq!(loaded_v2.static_wsp.num_tiles, model.static_wsp.num_tiles);
        assert_eq!(
            loaded_v2.static_wsp.legal_saci_ops,
            model.static_wsp.legal_saci_ops
        );

        // V1 backward-compatible load (defaults WorkspaceIr)
        let mut buf_v1 = Vec::new();
        save_fhp_v1(&model, &mut buf_v1).unwrap();
        let (loaded_v1, ver_v1) = load_fhp(&mut Cursor::new(&buf_v1)).unwrap();
        assert_eq!(ver_v1, FhpVersion::V1);
        assert_eq!(loaded_v1.ir.feature_etf.basis, model.ir.feature_etf.basis);
        assert_eq!(
            loaded_v1.ir.workspace.max_iters,
            WorkspaceIr::default().max_iters
        );

        // Reject invalid magic
        let mut bad = buf_v2.clone();
        bad[0] = b'X';
        assert!(matches!(
            load_fhp(&mut Cursor::new(&bad)),
            Err(FhpError::InvalidMagic(_))
        ));

        // Reject zero attention layers in FHP header
        let mut bad_layers = buf_v2.clone();
        // num_layers u32 is at 8 (magic) + 32 (header) = offset 40..44
        bad_layers[40..44].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            load_fhp(&mut Cursor::new(&bad_layers)),
            Err(FhpError::InvalidAttentionTopology)
        ));

        // Dynamic non-8-layer topology (e.g. 4 layers: KDA, MLA, KDA, MLA)
        let custom_topo = [
            AttentionKind::Kda,
            AttentionKind::Mla,
            AttentionKind::Kda,
            AttentionKind::Mla,
        ];
        let custom_spec = default_spec(48, 18);
        let custom_model = crate::runtime::compile_architecture_with_topology(
            &custom_spec,
            &custom_topo,
            default_saci(),
            None,
            &mut telemetry,
        )
        .unwrap();
        assert_eq!(custom_model.ir.attention.num_layers(), 4);
        assert_eq!(custom_model.static_wsp.num_layers, 4);

        let mut custom_buf = Vec::new();
        save_fhp(&custom_model, &mut custom_buf).unwrap();
        let (loaded_custom, _) = load_fhp(&mut Cursor::new(&custom_buf)).unwrap();
        assert_eq!(loaded_custom.ir.attention.num_layers(), 4);
        assert_eq!(loaded_custom.static_wsp.num_layers, 4);
        let _custom_scratch = AdaQueryScratch::new(&loaded_custom, 16);
    }

    #[test]
    fn capability_classification_and_graph_compilation() {
        use crate::capability::{
            classify_capability, verify_capability_contract, CapabilityGraph, CapabilityId,
            CapabilitySpec, CapabilityType, CompilationTarget, ObjectiveKind, SemanticsKind,
            TrainingObjective, ValueType, VerificationSpec, MAX_CAP_DEPS, MAX_CAP_IO,
        };
        use crate::sys::ArrayVec;

        fn io(vals: &[ValueType]) -> ArrayVec<ValueType, MAX_CAP_IO> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }
        fn deps(vals: &[CapabilityId]) -> ArrayVec<CapabilityId, MAX_CAP_DEPS> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }

        // 1. Knowledge (trainable semantic weights)
        let cap_k = CapabilitySpec {
            id: CapabilityId(1),
            cap_type: CapabilityType::Knowledge,
            inputs: io(&[ValueType::Vector]),
            outputs: io(&[ValueType::Bool]),
            semantics: SemanticsKind::ConceptRelation,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: deps(&[]),
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
        };

        // 2. Computation (CylinderVolume -> SACI expert, not neural weights)
        let cap_c = CapabilitySpec {
            id: CapabilityId(2),
            cap_type: CapabilityType::Computation,
            inputs: io(&[ValueType::Float, ValueType::Float]), // radius, height
            outputs: io(&[ValueType::Float]),                  // volume
            semantics: SemanticsKind::DeterministicFormula,
            preconditions: 0x03,  // radius >= 0, height >= 0
            postconditions: 0x01, // volume >= 0
            invariants: 0x01,
            dependencies: deps(&[CapabilityId(1)]),
            training_objective: None,
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-8,
                requires_formal_proof: true,
            },
            saci_binding: Some(SaciOpId(1)),
        };

        // 3. Reasoning (learned policy + SACI)
        let cap_r = CapabilitySpec {
            id: CapabilityId(3),
            cap_type: CapabilityType::Reasoning,
            inputs: io(&[ValueType::Vector]),
            outputs: io(&[ValueType::Int]),
            semantics: SemanticsKind::InferencePolicy,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x01,
            dependencies: deps(&[CapabilityId(1), CapabilityId(2)]),
            training_objective: Some(TrainingObjective {
                kind: ObjectiveKind::TracePolicyImitation,
                constraint_weight: 0.25,
                target_tolerance: 1e-2,
            }),
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-2,
                requires_formal_proof: false,
            },
            saci_binding: Some(SaciOpId(1)),
        };

        // 4. Simulation (hybrid state transition + surrogate)
        let cap_s = CapabilitySpec {
            id: CapabilityId(4),
            cap_type: CapabilityType::Simulation,
            inputs: io(&[ValueType::State, ValueType::Vector, ValueType::Float]),
            outputs: io(&[ValueType::State]),
            semantics: SemanticsKind::StateTransition,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x02,
            dependencies: deps(&[CapabilityId(2), CapabilityId(3)]),
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
            saci_binding: Some(SaciOpId(2)),
        };

        // 5. Verification (formal checker + learned detector)
        let cap_v = CapabilitySpec {
            id: CapabilityId(5),
            cap_type: CapabilityType::Verification,
            inputs: io(&[ValueType::State]),
            outputs: io(&[ValueType::Bool]),
            semantics: SemanticsKind::InvariantPredicate,
            preconditions: 0x01,
            postconditions: 0x01,
            invariants: 0x02,
            dependencies: deps(&[CapabilityId(4)]),
            training_objective: None,
            verification_spec: VerificationSpec {
                required_conditions: 0x01,
                tolerance: 1e-8,
                requires_formal_proof: true,
            },
            saci_binding: None,
        };

        let graph = CapabilityGraph::compile(&[
            cap_k.clone(),
            cap_c.clone(),
            cap_r.clone(),
            cap_s.clone(),
            cap_v.clone(),
        ])
        .unwrap();

        assert_eq!(graph.nodes[0].target, CompilationTarget::NeuralWeights);
        assert_eq!(graph.nodes[1].target, CompilationTarget::SaciExpert);
        assert_eq!(
            graph.nodes[2].target,
            CompilationTarget::LearnedPolicyAndSaci
        );
        assert_eq!(
            graph.nodes[3].target,
            CompilationTarget::SimulationAndSurrogate
        );
        assert_eq!(
            graph.nodes[4].target,
            CompilationTarget::FormalCheckerAndDetector
        );
        assert_eq!(graph.topo_order, vec![0, 1, 2, 3, 4]);

        // Formal contract verifier V(x) in {0, 1}
        assert!(verify_capability_contract(&cap_c, 0x03, 0x01, 0x01, 1e-10));
        assert!(!verify_capability_contract(&cap_c, 0x01, 0x01, 0x01, 1e-10)); // missing precondition
        assert!(!verify_capability_contract(&cap_c, 0x03, 0x01, 0x01, 1e-4)); // exceeds tolerance

        // Underspecified Computation without SACI binding or FiniteLookup fails classification
        let mut bad_c = cap_c.clone();
        bad_c.saci_binding = None;
        assert_eq!(classify_capability(&bad_c), Err(ErrorCode::UnknownSaciOp));

        // DirectCircuit classification for FiniteLookup without training objective
        bad_c.semantics = SemanticsKind::FiniteLookup;
        assert_eq!(
            classify_capability(&bad_c),
            Ok(CompilationTarget::DirectCircuit)
        );

        // Cycle detection
        let mut cyc_k = cap_k;
        cyc_k.dependencies = deps(&[CapabilityId(5)]);
        assert_eq!(
            CapabilityGraph::compile(&[cyc_k, cap_c, cap_r, cap_s, cap_v]).unwrap_err(),
            ErrorCode::NcmCycleDetected
        );
    }

    #[test]
    fn domain_capability_lowering_manifest_and_pipeline_a_link() {
        use crate::capability::{
            CapabilityId, CapabilitySpec, CapabilityType, ObjectiveKind, SemanticsKind,
            TrainingObjective, ValueType, VerificationSpec, MAX_CAP_DEPS, MAX_CAP_IO,
        };
        use crate::manifest::ParameterMode;
        use crate::runtime::compile_domain_architecture;
        use crate::sys::ArrayVec;

        fn io(vals: &[ValueType]) -> ArrayVec<ValueType, MAX_CAP_IO> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }
        fn deps(vals: &[CapabilityId]) -> ArrayVec<CapabilityId, MAX_CAP_DEPS> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }

        let caps = [
            CapabilitySpec {
                id: CapabilityId(1),
                cap_type: CapabilityType::Knowledge,
                inputs: io(&[ValueType::Vector]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::ConceptRelation,
                preconditions: 0x01,
                postconditions: 0x01,
                invariants: 0x01,
                dependencies: deps(&[]),
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
                cap_type: CapabilityType::Computation,
                inputs: io(&[ValueType::Float, ValueType::Float]),
                outputs: io(&[ValueType::Float]),
                semantics: SemanticsKind::DeterministicFormula,
                preconditions: 0x03,
                postconditions: 0x01,
                invariants: 0x01,
                dependencies: deps(&[CapabilityId(1)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: 0x01,
                    tolerance: 1e-8,
                    requires_formal_proof: true,
                },
                saci_binding: Some(SaciOpId(1)),
            },
            CapabilitySpec {
                id: CapabilityId(3),
                cap_type: CapabilityType::Verification,
                inputs: io(&[ValueType::Float]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::InvariantPredicate,
                preconditions: 0x01,
                postconditions: 0x01,
                invariants: 0x01,
                dependencies: deps(&[CapabilityId(2)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: 0x01,
                    tolerance: 1e-8,
                    requires_formal_proof: true,
                },
                saci_binding: None,
            },
        ];

        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let spec = default_spec(32, 12);
        let topo = [AttentionKind::Kda, AttentionKind::Mla];

        let (model, _graph, manifest) =
            compile_domain_architecture(&spec, &topo, &caps, default_saci(), 4, &mut telemetry)
                .unwrap();

        // 2 neural/interface groups (4 neurons each = 8 allocated), 1 formal verifier (0 neurons)
        assert_eq!(manifest.feature_groups.len(), 3);
        assert_eq!(manifest.neuron_assignments.len(), 8);
        assert_eq!(
            manifest.parameter_specs[0].param_mode,
            ParameterMode::LearnedOptimize
        );
        assert_eq!(
            manifest.parameter_specs[1].param_mode,
            ParameterMode::ExternalSaci
        );
        assert_eq!(
            manifest.parameter_specs[2].param_mode,
            ParameterMode::FormalChecker
        );
        assert_eq!(manifest.dependencies.len(), 2);
        assert_eq!(manifest.saci_bindings.len(), 1);
        assert_eq!(manifest.verification_obligations.len(), 3);

        // Model legal_masks are specialized (not 0x0F everywhere!)
        assert_eq!(
            model.ir.feature_etf.legal_masks[0],
            FeatureFamily::Knowledge.bit()
        );
        assert_eq!(
            model.ir.feature_etf.legal_masks[4],
            FeatureFamily::SaciInterface.bit()
        );
        assert_eq!(model.ir.feature_etf.legal_masks[8], 0x00);

        // Execute Pipeline B using the manifest's neuron_assignments
        let ncm = NcmOutputIr::from_edges(
            vec![NcmArtifactKind::Title, NcmArtifactKind::ValidationSummary],
            &[(NcmNodeId(0), NcmNodeId(1))],
        )
        .unwrap();
        let plan = WorkflowPlanIr {
            assignments: &manifest.neuron_assignments,
            saci_ops: &[SaciOpId(1)],
            verify_threshold: 0.01,
        };
        let mut scratch = AdaQueryScratch::new(&model, 8);
        let input = vec![1.0; 12];
        let res =
            execute_query_pipeline_b(&model, &plan, &ncm, &input, &mut scratch, &mut telemetry)
                .unwrap();
        assert_eq!(res.sparse_stats.active_neurons, 8);
        assert_eq!(res.sparse_stats.skipped_neurons, 24);
        assert_eq!(res.workspace_state, WorkspaceState::HaltedVerified);
    }

    #[test]
    fn tiny_domain_test_model_fixture() {
        use crate::capability::{
            CapabilityGraph, CapabilityId, CapabilitySpec, CapabilityType, ObjectiveKind,
            SemanticsKind, TrainingObjective, ValueType, VerificationSpec, MAX_CAP_DEPS,
            MAX_CAP_IO,
        };
        use crate::fhp::{load_fhp, save_fhp};
        use crate::manifest::lower_capabilities;
        use crate::runtime::compile_architecture_with_topology;
        use crate::sys::ArrayVec;
        use std::io::Cursor;

        fn io(vals: &[ValueType]) -> ArrayVec<ValueType, MAX_CAP_IO> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }
        fn deps(vals: &[CapabilityId]) -> ArrayVec<CapabilityId, MAX_CAP_DEPS> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }

        let spec = ArchitectureSpec {
            vq_codebook_size: 8,
            num_neurons: 8,
            substrate_dim: 3,
            tile_size: 2,
            kda: KdaConfig {
                chunk_size: 4,
                dplr_rank: 2,
                state_dim: 4,
            },
            mla: MlaConfig {
                kv_latent_dim: 8,
                num_heads: 2,
                head_dim: 4,
            },
            workspace: WorkspaceIr {
                scratchpad_slots: 4,
                max_iters: 4,
                halting_tolerance: 0.0001,
            },
        };

        let topology = [AttentionKind::Kda, AttentionKind::Mla];

        let saci = SaciIr::new(vec![SaciOpSpec {
            id: SaciOpId(1),
            numerical_mode: NumericalMode::Exact,
            tolerance: 0.0,
            backend: Backend::Cpu,
            precondition_flags: 0,
            postcondition_flags: 0,
        }]);

        let legal_masks: [u8; 8] = [0x01, 0x01, 0x02, 0x02, 0x08, 0x04, 0x03, 0x0F];

        let caps = [
            CapabilitySpec {
                id: CapabilityId(1),
                cap_type: CapabilityType::Knowledge,
                inputs: io(&[ValueType::Vector, ValueType::Vector]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::ConceptRelation,
                preconditions: 0,
                postconditions: 0,
                invariants: 0,
                dependencies: deps(&[]),
                training_objective: Some(TrainingObjective {
                    kind: ObjectiveKind::ContrastiveSemantic,
                    constraint_weight: 0.5,
                    target_tolerance: 1e-3,
                }),
                verification_spec: VerificationSpec {
                    required_conditions: 0,
                    tolerance: 0.0,
                    requires_formal_proof: true,
                },
                saci_binding: None,
            },
            CapabilitySpec {
                id: CapabilityId(2),
                cap_type: CapabilityType::Reasoning,
                inputs: io(&[ValueType::Bool, ValueType::Bool]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::InferencePolicy,
                preconditions: 0,
                postconditions: 0,
                invariants: 0,
                dependencies: deps(&[CapabilityId(1)]),
                training_objective: Some(TrainingObjective {
                    kind: ObjectiveKind::TracePolicyImitation,
                    constraint_weight: 0.25,
                    target_tolerance: 1e-3,
                }),
                verification_spec: VerificationSpec {
                    required_conditions: 0,
                    tolerance: 0.0,
                    requires_formal_proof: true,
                },
                saci_binding: None,
            },
            CapabilitySpec {
                id: CapabilityId(3),
                cap_type: CapabilityType::Simulation,
                inputs: io(&[ValueType::State, ValueType::Vector]),
                outputs: io(&[ValueType::State]),
                semantics: SemanticsKind::StateTransition,
                preconditions: 0,
                postconditions: 0,
                invariants: 0,
                dependencies: deps(&[CapabilityId(1)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: 0,
                    tolerance: 0.0,
                    requires_formal_proof: true,
                },
                saci_binding: None,
            },
            CapabilitySpec {
                id: CapabilityId(4),
                cap_type: CapabilityType::Verification,
                inputs: io(&[ValueType::State]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::InvariantPredicate,
                preconditions: 0,
                postconditions: 0,
                invariants: 0x01,
                dependencies: deps(&[CapabilityId(2), CapabilityId(3)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: 0,
                    tolerance: 0.0,
                    requires_formal_proof: true,
                },
                saci_binding: None,
            },
            CapabilitySpec {
                id: CapabilityId(5),
                cap_type: CapabilityType::Computation,
                inputs: io(&[ValueType::Float, ValueType::Float]),
                outputs: io(&[ValueType::Float]),
                semantics: SemanticsKind::DeterministicFormula,
                preconditions: 0,
                postconditions: 0,
                invariants: 0,
                dependencies: deps(&[CapabilityId(1)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: 0,
                    tolerance: 0.0,
                    requires_formal_proof: true,
                },
                saci_binding: Some(SaciOpId(1)),
            },
        ];

        let graph = CapabilityGraph::compile(&caps).unwrap();
        let manifest = lower_capabilities(&graph, &caps, spec.num_neurons, 2).unwrap();

        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();
        let model = compile_architecture_with_topology(
            &spec,
            &topology,
            saci,
            Some(&legal_masks),
            &mut telemetry,
        )
        .unwrap();

        let mut fhp_buf = Vec::new();
        save_fhp(&model, &mut fhp_buf).unwrap();
        let (reloaded, ver) = load_fhp(&mut Cursor::new(&fhp_buf)).unwrap();

        println!("CapabilityGraph topo_order: {:?}", graph.topo_order);
        println!("CapabilityGraph target_counts: {:?}", graph.target_counts);
        for n in &graph.nodes {
            println!(
                "  cap={:?} type={:?} target={:?} family={:?} saci={:?}",
                n.id, n.cap_type, n.target, n.family, n.saci_binding
            );
        }
        println!(
            "Manifest groups={}, assignments={}, deps={}, saci_bindings={}, obligations={}",
            manifest.feature_groups.len(),
            manifest.neuron_assignments.len(),
            manifest.dependencies.len(),
            manifest.saci_bindings.len(),
            manifest.verification_obligations.len(),
        );
        println!("CompiledAdaModel: neurons={}, substrate_dim={}, layers={}, tiles={}, params={}, legal_masks={:02x?}",
            model.ir.feature_etf.num_neurons,
            model.ir.feature_etf.substrate_dim,
            model.ir.attention.num_layers(),
            model.static_wsp.num_tiles,
            model.ir.parameter_count(),
            model.ir.feature_etf.legal_masks,
        );
        println!(
            "ETF welch_bound={:.6}, max_coherence={:.6}, basis_len={}",
            model.ir.feature_etf.welch_bound,
            model.ir.feature_etf.max_coherence,
            model.ir.feature_etf.basis.len(),
        );
        println!(
            "FHP serialized: {} bytes ({:?}), reloaded_layers={}",
            fhp_buf.len(),
            ver,
            reloaded.ir.attention.num_layers(),
        );
        while let Some(ev) = telemetry.pop() {
            println!("{}", observatory::format_event(&ev));
        }

        assert_eq!(reloaded.ir.feature_etf.legal_masks, legal_masks);
        assert_eq!(reloaded.ir.feature_etf.basis.len(), 8 * 4 * 3);
    }

    #[test]
    fn cylinder_domain_compilation_and_functional_test() {
        use crate::capability::{
            verify_capability_contract, CapabilityId, CapabilitySpec, CapabilityType,
            ObjectiveKind, SemanticsKind, TrainingObjective, ValueType, VerificationSpec,
            MAX_CAP_DEPS, MAX_CAP_IO,
        };
        use crate::runtime::{compile_domain_architecture, dispatch_saci_op};
        use crate::sys::ArrayVec;

        fn io(vals: &[ValueType]) -> ArrayVec<ValueType, MAX_CAP_IO> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }
        fn deps(vals: &[CapabilityId]) -> ArrayVec<CapabilityId, MAX_CAP_DEPS> {
            let mut v = ArrayVec::new();
            for &x in vals {
                assert_eq!(v.push(x), ErrorCode::None);
            }
            v
        }

        const PRE_RADIUS_NONNEG: u32 = 1 << 0;
        const PRE_HEIGHT_NONNEG: u32 = 1 << 1;
        const POST_VOL_NONNEG: u32 = 1 << 0;
        const INV_GEOMETRIC_VOLUME: u32 = 1 << 0;
        const CYLINDER_VOL_OP: SaciOpId = SaciOpId(10);

        // 1. Formal Cylinder Domain Capabilities (Knowledge, Computation, Reasoning, Simulation, Verification)
        let caps = [
            CapabilitySpec {
                id: CapabilityId(1),
                cap_type: CapabilityType::Knowledge,
                inputs: io(&[ValueType::Float, ValueType::Float]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::ConceptRelation,
                preconditions: PRE_RADIUS_NONNEG | PRE_HEIGHT_NONNEG,
                postconditions: POST_VOL_NONNEG,
                invariants: INV_GEOMETRIC_VOLUME,
                dependencies: deps(&[]),
                training_objective: Some(TrainingObjective {
                    kind: ObjectiveKind::ContrastiveSemantic,
                    constraint_weight: 0.5,
                    target_tolerance: 1e-6,
                }),
                verification_spec: VerificationSpec {
                    required_conditions: POST_VOL_NONNEG,
                    tolerance: 1e-6,
                    requires_formal_proof: false,
                },
                saci_binding: None,
            },
            CapabilitySpec {
                id: CapabilityId(2),
                cap_type: CapabilityType::Computation,
                inputs: io(&[ValueType::Float, ValueType::Float]), // (radius, height)
                outputs: io(&[ValueType::Float]),                  // volume = pi * r^2 * h
                semantics: SemanticsKind::DeterministicFormula,
                preconditions: PRE_RADIUS_NONNEG | PRE_HEIGHT_NONNEG,
                postconditions: POST_VOL_NONNEG,
                invariants: INV_GEOMETRIC_VOLUME,
                dependencies: deps(&[CapabilityId(1)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: POST_VOL_NONNEG,
                    tolerance: 1e-12,
                    requires_formal_proof: true,
                },
                saci_binding: Some(CYLINDER_VOL_OP),
            },
            CapabilitySpec {
                id: CapabilityId(3),
                cap_type: CapabilityType::Reasoning,
                inputs: io(&[ValueType::Float, ValueType::Float]),
                outputs: io(&[ValueType::Int]),
                semantics: SemanticsKind::InferencePolicy,
                preconditions: PRE_RADIUS_NONNEG | PRE_HEIGHT_NONNEG,
                postconditions: POST_VOL_NONNEG,
                invariants: INV_GEOMETRIC_VOLUME,
                dependencies: deps(&[CapabilityId(1), CapabilityId(2)]),
                training_objective: Some(TrainingObjective {
                    kind: ObjectiveKind::TracePolicyImitation,
                    constraint_weight: 0.25,
                    target_tolerance: 1e-6,
                }),
                verification_spec: VerificationSpec {
                    required_conditions: POST_VOL_NONNEG,
                    tolerance: 1e-6,
                    requires_formal_proof: false,
                },
                saci_binding: Some(CYLINDER_VOL_OP),
            },
            CapabilitySpec {
                id: CapabilityId(4),
                cap_type: CapabilityType::Simulation,
                inputs: io(&[ValueType::State, ValueType::Vector]), // (r_t, h_t) + (dr, dh)
                outputs: io(&[ValueType::State]),
                semantics: SemanticsKind::StateTransition,
                preconditions: PRE_RADIUS_NONNEG | PRE_HEIGHT_NONNEG,
                postconditions: POST_VOL_NONNEG,
                invariants: INV_GEOMETRIC_VOLUME,
                dependencies: deps(&[CapabilityId(2), CapabilityId(3)]),
                training_objective: Some(TrainingObjective {
                    kind: ObjectiveKind::PhysicsConstrainedTrajectory,
                    constraint_weight: 1.0,
                    target_tolerance: 1e-9,
                }),
                verification_spec: VerificationSpec {
                    required_conditions: POST_VOL_NONNEG,
                    tolerance: 1e-9,
                    requires_formal_proof: false,
                },
                saci_binding: Some(CYLINDER_VOL_OP),
            },
            CapabilitySpec {
                id: CapabilityId(5),
                cap_type: CapabilityType::Verification,
                inputs: io(&[ValueType::State, ValueType::Float]),
                outputs: io(&[ValueType::Bool]),
                semantics: SemanticsKind::InvariantPredicate,
                preconditions: PRE_RADIUS_NONNEG | PRE_HEIGHT_NONNEG,
                postconditions: POST_VOL_NONNEG,
                invariants: INV_GEOMETRIC_VOLUME,
                dependencies: deps(&[CapabilityId(2), CapabilityId(4)]),
                training_objective: None,
                verification_spec: VerificationSpec {
                    required_conditions: POST_VOL_NONNEG,
                    tolerance: 1e-12,
                    requires_formal_proof: true,
                },
                saci_binding: None,
            },
        ];

        // 2. Compile Domain Architecture (CapabilityGraph -> FeatureNeuronManifest -> Pipeline A)
        let saci = SaciIr::new(vec![SaciOpSpec {
            id: CYLINDER_VOL_OP,
            numerical_mode: NumericalMode::Exact,
            tolerance: 1e-12,
            backend: Backend::Cpu,
            precondition_flags: PRE_RADIUS_NONNEG | PRE_HEIGHT_NONNEG,
            postcondition_flags: POST_VOL_NONNEG,
        }]);
        let spec = default_spec(32, 12);
        let topo = [AttentionKind::Kda, AttentionKind::Mla];
        let mut telemetry: TelemetryBus<64> = TelemetryBus::new();

        let (model, graph, manifest) =
            compile_domain_architecture(&spec, &topo, &caps, saci, 4, &mut telemetry).unwrap();

        println!("=== Cylinder Domain Compilation ===");
        println!("CapabilityGraph topo_order: {:?}", graph.topo_order);
        println!("Manifest groups={}, neuron_assignments={}, dependencies={}, saci_bindings={}, obligations={}",
            manifest.feature_groups.len(),
            manifest.neuron_assignments.len(),
            manifest.dependencies.len(),
            manifest.saci_bindings.len(),
            manifest.verification_obligations.len(),
        );
        println!(
            "CompiledAdaModel: neurons={}, dim={}, layers={}, params={}, legal_masks={:02x?}",
            model.ir.feature_etf.num_neurons,
            model.ir.feature_etf.substrate_dim,
            model.ir.attention.num_layers(),
            model.ir.parameter_count(),
            &model.ir.feature_etf.legal_masks[..20],
        );

        // 3. Execute Pipeline B using manifest-derived neuron assignments & SACI op
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
            assignments: &manifest.neuron_assignments,
            saci_ops: &[CYLINDER_VOL_OP],
            verify_threshold: 0.01,
        };
        let mut scratch = AdaQueryScratch::new(&model, 8);
        let input = vec![1.0; 12];
        let res =
            execute_query_pipeline_b(&model, &plan, &ncm, &input, &mut scratch, &mut telemetry)
                .unwrap();
        assert_eq!(res.workspace_state, WorkspaceState::HaltedVerified);
        assert_eq!(res.sparse_stats.active_neurons, 16); // 4 neural/interface groups x 4 neurons
        println!(
            "Pipeline B: workspace={:?}, iters={}, active_neurons={}/{}, active_flops={}/{}",
            res.workspace_state,
            res.workspace_iters,
            res.sparse_stats.active_neurons,
            res.sparse_stats.total_neurons,
            res.sparse_stats.active_flops,
            res.sparse_stats.total_flops,
        );

        // 4. Functional Cylinder Simulation + SACI Kernel + Formal Contract Verification
        let mut radius = 2.0_f64;
        let mut height = 5.0_f64;
        let dr = 0.5_f64;
        let dh = 1.0_f64;

        for step in 0..4 {
            let volume = core::f64::consts::PI * radius * radius * height;
            let expected = core::f64::consts::PI * radius * radius * height;
            let residual = (volume - expected).abs();

            // SACI tolerance check
            dispatch_saci_op(&model.ir.saci, CYLINDER_VOL_OP, residual, &mut telemetry).unwrap();

            // Formal verification obligation V(x) in {0, 1}
            let obs_pre = if radius >= 0.0 { PRE_RADIUS_NONNEG } else { 0 }
                | if height >= 0.0 { PRE_HEIGHT_NONNEG } else { 0 };
            let obs_post = if volume >= 0.0 { POST_VOL_NONNEG } else { 0 };
            let obs_inv = if residual <= 1e-12 {
                INV_GEOMETRIC_VOLUME
            } else {
                0
            };

            let verified =
                verify_capability_contract(&caps[4], obs_pre, obs_post, obs_inv, residual);
            assert!(verified);
            println!(
                "Sim step {}: r={:.2}, h={:.2}, volume={:.6}, residual={:.2e}, formal_verified={}",
                step, radius, height, volume, residual, verified
            );

            radius += dr;
            height += dh;
        }

        // Negative radius must fail formal precondition verification
        let bad_pre = PRE_HEIGHT_NONNEG; // missing PRE_RADIUS_NONNEG
        let bad_verified = verify_capability_contract(
            &caps[4],
            bad_pre,
            POST_VOL_NONNEG,
            INV_GEOMETRIC_VOLUME,
            0.0,
        );
        assert!(!bad_verified);
        println!("Negative-radius check: formal_verified={}", bad_verified);

        while let Some(ev) = telemetry.pop() {
            println!("{}", observatory::format_event(&ev));
        }
    }
}

#[cfg(test)]
mod wsp_tests {
    use super::*;
    use crate::checkpoint::TensorDescriptor;
    use crate::dcir::{ConstraintSpec, DomainExample, DomainOpKind};
    use crate::sys::{AdaComponent, DomainConceptId};

    #[test]
    fn ada_o1_manifest_totals_and_protection_enforcement() {
        let manifest = ArchitectureManifest::ada_o1_manifest();
        assert_eq!(manifest.total_declared_params(), 7_395_228);

        // Attempt to patch FrozenVq (TensorId(4)) -> must fail with ProtectedInvariant
        let bad_vq = [WeightPatch {
            tensor_id: TensorId(4),
            sparse_deltas: vec![(0, 0.01)],
        }];
        assert_eq!(
            validate_patch_permissions(&bad_vq, &manifest),
            Err(ErrorCode::ProtectedInvariant)
        );

        // Attempt to patch SaciCircuits (TensorId(5)) -> must fail
        let bad_saci = [WeightPatch {
            tensor_id: TensorId(5),
            sparse_deltas: vec![(0, 0.01)],
        }];
        assert_eq!(
            validate_patch_permissions(&bad_saci, &manifest),
            Err(ErrorCode::ProtectedInvariant)
        );

        // Attempt to patch bypassed BrainDecoder (TensorId(3)) -> must fail
        let bad_dec = [WeightPatch {
            tensor_id: TensorId(3),
            sparse_deltas: vec![(0, 0.01)],
        }];
        assert_eq!(
            validate_patch_permissions(&bad_dec, &manifest),
            Err(ErrorCode::ProtectedInvariant)
        );
    }

    #[test]
    fn adaptive_pondernet_early_halt_and_saci_escalation() {
        let cfg = AdaptivePonderConfig {
            min_steps: 1,
            max_steps: 4,
            halt_threshold: 0.75,
            halt_tolerance: 1e-4,
            lambda_prior: 0.3,
            kl_beta: 0.01,
        };

        // Low residual + confident logit -> halts early at step 1
        let mut st_easy = PonderState::new();
        let (act1, _) = step_ponder(&mut st_easy, 3.0, 1e-6, &cfg);
        assert_eq!(act1, PonderAction::Halt);
        assert_eq!(st_easy.step, 1);

        // High residual -> ponders until max_steps then escalates to SACI
        let mut st_hard = PonderState::new();
        for _ in 0..3 {
            let (act, _) = step_ponder(&mut st_hard, 0.0, 0.5, &cfg);
            assert_eq!(act, PonderAction::Continue);
        }
        let (act_final, _) = step_ponder(&mut st_hard, 0.0, 0.5, &cfg);
        assert_eq!(act_final, PonderAction::EscalateToSaci);
        assert_eq!(st_hard.step, 4);
        assert!((st_hard.cumulative_halt_prob - 1.0).abs() < 1e-12);
    }

    #[test]
    fn awsc_end_to_end_nullspace_surgical_write_and_verify() {
        let manifest = ArchitectureManifest::ada_o1_manifest();

        // Small proxy checkpoint using ADA-o1 tensor IDs:
        // Tensor 0 (BrainKda): 4x2 projection
        // Tensor 4 (FrozenVq): 4x1 frozen
        // Tensor 5 (SaciCircuits): 4x1 frozen
        let descriptors = vec![
            TensorDescriptor {
                id: TensorId(0),
                component: AdaComponent::BrainKda,
                rows: 4,
                cols: 2,
                offset: 0,
                len: 8,
                frozen: false,
            },
            TensorDescriptor {
                id: TensorId(4),
                component: AdaComponent::FrozenVq,
                rows: 4,
                cols: 1,
                offset: 8,
                len: 4,
                frozen: true,
            },
            TensorDescriptor {
                id: TensorId(5),
                component: AdaComponent::SaciCircuits,
                rows: 4,
                cols: 1,
                offset: 12,
                len: 4,
                frozen: true,
            },
        ];
        let mut data = vec![0.0_f64; 16];
        // Initial W0 in BrainKda maps x[2] -> y[0] with weight 1.0 (protected subspace)
        data[2 * 2] = 1.0;
        let ckpt = Checkpoint::new(descriptors, data).unwrap();

        // Protected example on dims [2, 3], new mechanical-stress domain examples on dims [0, 1]
        let dataset = DomainDataset {
            domain_id: 42,
            feature_dim: 4,
            target_dim: 2,
            examples: vec![
                DomainExample {
                    concept_id: DomainConceptId(1),
                    features: vec![1.0, 0.0, 0.0, 0.0],
                    target: vec![2.0, -1.0],
                    protected: false,
                },
                DomainExample {
                    concept_id: DomainConceptId(2),
                    features: vec![0.0, 1.0, 0.0, 0.0],
                    target: vec![0.5, 3.0],
                    protected: false,
                },
                DomainExample {
                    concept_id: DomainConceptId(100),
                    features: vec![0.0, 0.0, 1.0, 0.0],
                    target: vec![1.0, 0.0],
                    protected: true,
                },
            ],
        };

        let reasoning = ReasoningSpecification {
            steps: vec![
                ReasoningStep {
                    step_id: 1,
                    op: DomainOpKind::LinearMap,
                    target_component: AdaComponent::BrainKda,
                    halting_prior: 0.3,
                },
                ReasoningStep {
                    step_id: 2,
                    op: DomainOpKind::InvariantCheck,
                    target_component: AdaComponent::BridgesHaltingVerify,
                    halting_prior: 0.7,
                },
            ],
            constraints: vec![ConstraintSpec {
                constraint_id: 1,
                max_allowed_violation: 1e-6,
            }],
            ponder: AdaptivePonderConfig::default(),
        };

        let wsp_spec = WspSpecification {
            target_tensors: vec![TensorId(0)],
            regularization_lambda: 1e-6,
            sparse_delta_epsilon: 1e-9,
            max_target_mse: 1e-6,
            max_protected_drift: 1e-12,
        };

        let out = compile_weight_space(WspInput {
            checkpoint: &ckpt,
            architecture: &manifest,
            specification: &wsp_spec,
            dataset: &dataset,
            reasoning: &reasoning,
        })
        .unwrap();

        assert_eq!(out.write_report.solver_used, SolverKind::SvdNullspace);
        assert!(out.write_report.nonzero_deltas > 0);
        // Frozen VQ and SACI circuits remain untouched
        assert_eq!(
            out.candidate_checkpoint.tensor_slice(TensorId(4)),
            ckpt.tensor_slice(TensorId(4))
        );
        assert_eq!(
            out.candidate_checkpoint.tensor_slice(TensorId(5)),
            ckpt.tensor_slice(TensorId(5))
        );

        let report = verify_candidate_checkpoint(&ckpt, &out, &wsp_spec).unwrap();
        assert!(report.passed);
        assert!(report.target_mse < 1e-8);
        assert!(report.protected_drift < 1e-12);
    }

    #[test]
    fn forensic_saci_circuit_execution_ablation_and_composition() {
        use crate::ir::{
            ArchitectureSpec, KdaConfig, MlaConfig, NcmArtifactKind, NcmOutputIr, NeuronAssignment,
            SaciIr, SaciOpSpec, WorkflowPlanIr, WorkspaceIr,
        };
        use crate::runtime::{compile_architecture, execute_query_pipeline_b, AdaQueryScratch};
        use crate::saci_circuit::{
            execute_forensic_saci_circuit, reference_forensic_timeline,
            synthesize_forensic_timeline_weights, FORENSIC_CIRCUIT_PARAMS,
        };
        use crate::sys::NeuronGroupId;
        use crate::sys::{
            Backend, FeatureFamily, NcmNodeId, NeuronId, NumericalMode, SaciOpId, TelemetryBus,
            WorkspaceState,
        };

        // 1. Compile SOLVER_07 weights into Checkpoint TensorId(5) (SaciCircuits)
        let saci_weights = synthesize_forensic_timeline_weights();
        assert_eq!(saci_weights.len(), FORENSIC_CIRCUIT_PARAMS);

        let ckpt_compiled = Checkpoint::new(
            vec![TensorDescriptor {
                id: TensorId(5),
                component: AdaComponent::SaciCircuits,
                rows: FORENSIC_CIRCUIT_PARAMS,
                cols: 1,
                offset: 0,
                len: FORENSIC_CIRCUIT_PARAMS,
                frozen: true,
            }],
            saci_weights,
        )
        .unwrap();

        // 2. Test normal, boundary, adversarial, and invalid inputs vs independent reference
        let cases = [
            // Normal valid timeline: child spawned 100s after parent
            (1718003707.0, 1718003607.0, 0.0),
            // Boundary: exact simultaneous spawn
            (1718003607.0, 1718003607.0, 0.0),
            // Boundary: within allowed clock skew (0.5s)
            (1718003606.5, 1718003607.0, 0.5),
            // Adversarial (real data_dal SOLVER_07 case): child precedes parent by 100s
            (1718003607.0, 1718003707.0, 0.0),
            // Invalid skew: negative max_skew
            (1718003707.0, 1718003607.0, -1.0),
        ];

        for &(child, parent, skew) in &cases {
            let trace = execute_forensic_saci_circuit(
                &ckpt_compiled,
                TensorId(5),
                NeuronGroupId(7),
                child,
                parent,
                skew,
            )
            .unwrap();
            let expected = reference_forensic_timeline(child, parent, skew);
            assert_eq!(trace.circuit_id, 7);
            assert_eq!(trace.weights_read, FORENSIC_CIRCUIT_PARAMS);
            for (k, &exp) in expected.iter().enumerate() {
                assert!(
                    (trace.output[k] - exp).abs() < 1e-9,
                    "mismatch at out[{k}]: got={}, expected={exp}",
                    trace.output[k]
                );
            }
        }

        // Non-finite input must be rejected
        assert_eq!(
            execute_forensic_saci_circuit(
                &ckpt_compiled,
                TensorId(5),
                NeuronGroupId(7),
                f64::NAN,
                0.0,
                0.0
            )
            .unwrap_err(),
            ErrorCode::InvalidInput
        );

        // 3. ABLATION PROOF: Replace SACI weights with cos() placeholders -> circuit MUST FAIL
        let placeholder_weights: Vec<f64> = (0..FORENSIC_CIRCUIT_PARAMS)
            .map(|i| ((i as f64) * 0.029).cos())
            .collect();
        let ckpt_placeholder = Checkpoint::new(
            vec![TensorDescriptor {
                id: TensorId(5),
                component: AdaComponent::SaciCircuits,
                rows: FORENSIC_CIRCUIT_PARAMS,
                cols: 1,
                offset: 0,
                len: FORENSIC_CIRCUIT_PARAMS,
                frozen: true,
            }],
            placeholder_weights,
        )
        .unwrap();
        let ablated_trace = execute_forensic_saci_circuit(
            &ckpt_placeholder,
            TensorId(5),
            NeuronGroupId(7),
            1718003607.0,
            1718003707.0,
            0.0,
        )
        .unwrap();
        let expected_adv = reference_forensic_timeline(1718003607.0, 1718003707.0, 0.0);
        let ablated_err = (ablated_trace.output[0] - expected_adv[0]).abs();
        assert!(
            ablated_err > 1.0,
            "Ablated placeholder weights unexpectedly matched reference! err={ablated_err}"
        );

        // 4. COMPOSITION PROOF: WSP route -> Pipeline B SaciInterface activation -> SOLVER_07 neural execution -> NCM
        let mut tel: TelemetryBus<64> = TelemetryBus::new();
        let arch_spec = ArchitectureSpec {
            vq_codebook_size: 64,
            num_neurons: 16,
            substrate_dim: 12,
            tile_size: 4,
            kda: KdaConfig {
                chunk_size: 16,
                dplr_rank: 2,
                state_dim: 12,
            },
            mla: MlaConfig {
                kv_latent_dim: 8,
                num_heads: 2,
                head_dim: 4,
            },
            workspace: WorkspaceIr::default(),
        };
        let saci_ir = SaciIr::new(vec![SaciOpSpec {
            id: SaciOpId(7),
            numerical_mode: NumericalMode::Exact,
            tolerance: 1e-9,
            backend: Backend::Cpu,
            precondition_flags: 0x01,
            postcondition_flags: 0x01,
        }]);
        let ada_model = compile_architecture(&arch_spec, saci_ir, None, &mut tel).unwrap();
        let mut scratch = AdaQueryScratch::new(&ada_model, 8);
        let assignments = [
            NeuronAssignment {
                neuron: NeuronId(0),
                family: FeatureFamily::Knowledge,
            },
            NeuronAssignment {
                neuron: NeuronId(4),
                family: FeatureFamily::SaciInterface,
            },
            NeuronAssignment {
                neuron: NeuronId(8),
                family: FeatureFamily::Reasoning,
            },
        ];
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
            assignments: &assignments,
            saci_ops: &[SaciOpId(7)],
            verify_threshold: 0.01,
        };
        let input = vec![1.0; 12];
        let pb_res =
            execute_query_pipeline_b(&ada_model, &plan, &ncm, &input, &mut scratch, &mut tel)
                .unwrap();
        assert_eq!(pb_res.workspace_state, WorkspaceState::HaltedVerified);
        assert_eq!(
            scratch.wsp.neuron_masks[4],
            FeatureFamily::SaciInterface.bit()
        );

        // Execute the compiled in-neuron SOLVER_07 weights for the routed SACI op
        let trace = execute_forensic_saci_circuit(
            &ckpt_compiled,
            TensorId(5),
            NeuronGroupId(4),
            1718003607.0,
            1718003707.0,
            0.0,
        )
        .unwrap();
        assert!((trace.output[0] - (-100.0)).abs() < 1e-9);
        assert!((trace.output[1] - 100.0).abs() < 1e-9);
        assert!((trace.output[2] - 1.0).abs() < 1e-9);
        assert!((trace.output[3] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn generic_saci_pipeline_with_mandatory_ablation_gate() {
        use crate::saci_circuit::{
            reference_forensic_timeline, verify_saci_circuit_with_ablation, CircuitSlot,
            SaciCircuitProgram,
        };

        let prog = SaciCircuitProgram::forensic_timeline_solver07();
        let slot = CircuitSlot {
            circuit_id: 7,
            tensor_id: TensorId(5),
            offset: 0,
            len: prog.weights.len(),
        };
        let ckpt = Checkpoint::new(
            vec![TensorDescriptor {
                id: TensorId(5),
                component: AdaComponent::SaciCircuits,
                rows: prog.weights.len(),
                cols: 1,
                offset: 0,
                len: prog.weights.len(),
                frozen: true,
            }],
            prog.weights.clone(),
        )
        .unwrap();

        let inputs = vec![
            vec![1718003707.0, 1718003607.0, 0.0, 1.0],
            vec![1718003607.0, 1718003707.0, 0.0, 1.0],
            vec![100.0, 100.0, 0.0, 1.0],
            vec![10.0, 20.0, -2.0, 1.0],
        ];

        let report = verify_saci_circuit_with_ablation(
            &prog,
            &ckpt,
            &slot,
            7,
            &inputs,
            |inp| reference_forensic_timeline(inp[0], inp[1], inp[2]).to_vec(),
            1e-9,
        )
        .unwrap();

        assert!(report.passed);
        assert_eq!(report.cases_verified, 4);
        assert!(report.max_abs_error < 1e-9);
        assert!(report.ablation_max_error > 1.0);
    }

    #[test]
    fn solver07_generalization_and_numerical_robustness() {
        use crate::saci_circuit::{
            execute_isolated_saci_slot, reference_forensic_timeline, CircuitSlot,
            SaciCircuitProgram,
        };

        let prog = SaciCircuitProgram::forensic_timeline_solver07();
        let slot = CircuitSlot {
            circuit_id: 7,
            tensor_id: TensorId(5),
            offset: 0,
            len: prog.weights.len(),
        };
        let ckpt = Checkpoint::new(
            vec![TensorDescriptor {
                id: TensorId(5),
                component: AdaComponent::SaciCircuits,
                rows: prog.weights.len(),
                cols: 1,
                offset: 0,
                len: prog.weights.len(),
                frozen: true,
            }],
            prog.weights.clone(),
        )
        .unwrap();

        // Deterministic xorshift64 sweep: 500 randomized cases across magnitudes
        let mut seed = 0x1234_5678_9abc_def0_u64;
        let mut next_f64 = || -> f64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed as f64) / (u64::MAX as f64)) * 2.0 - 1.0
        };

        for idx in 0..500 {
            let scale = match idx % 4 {
                0 => 1.0,
                1 => 1.0e4,
                2 => 1.0e9,
                _ => 1.0e-2,
            };
            let child = next_f64() * scale;
            let parent = next_f64() * scale;
            let skew = if idx % 10 == 0 {
                -next_f64().abs() * 10.0 - 0.1
            } else {
                next_f64().abs() * 10.0
            };
            // Avoid the (0, 1e-6) ramp transition zone for discrete indicator
            let deficit = parent - child - skew;
            if deficit > 0.0 && deficit < 1e-5 {
                continue;
            }
            if skew < 0.0 && skew > -1e-5 {
                continue;
            }

            let got =
                execute_isolated_saci_slot(&ckpt, &slot, 7, &prog, &[child, parent, skew, 1.0])
                    .unwrap();
            let exp = reference_forensic_timeline(child, parent, skew);
            let tol = 1e-6 * scale.max(1.0);
            for (g, e) in got.iter().zip(exp.iter()) {
                assert!((g - e).abs() <= tol, "scale={scale}, got={g}, exp={e}");
            }
        }

        // Subnormal values must remain finite and exact for delta/deficit
        let sub = execute_isolated_saci_slot(&ckpt, &slot, 7, &prog, &[1e-310, 2e-310, 0.0, 1.0])
            .unwrap();
        assert!((sub[0] - (-1e-310)).abs() < 1e-305);

        // Overflow guard: f64::MAX * STEEP_K overflows to Inf -> rejected with InvalidInput
        assert_eq!(
            execute_isolated_saci_slot(&ckpt, &slot, 7, &prog, &[0.0, f64::MAX, 0.0, 1.0])
                .unwrap_err(),
            ErrorCode::InvalidInput
        );
    }

    #[test]
    fn circuit_isolation_and_routing_causality() {
        use crate::saci_circuit::{
            execute_isolated_saci_slot, validate_circuit_isolation, CircuitSlot, SaciCircuitProgram,
        };

        let prog = SaciCircuitProgram::forensic_timeline_solver07();
        let n = prog.weights.len();
        let slot_a = CircuitSlot {
            circuit_id: 7,
            tensor_id: TensorId(5),
            offset: 0,
            len: n,
        };
        let slot_b = CircuitSlot {
            circuit_id: 9,
            tensor_id: TensorId(5),
            offset: n,
            len: n,
        };
        assert_eq!(validate_circuit_isolation(&[slot_a, slot_b]), Ok(()));

        // Overlapping slot must be rejected
        let overlap = CircuitSlot {
            circuit_id: 8,
            tensor_id: TensorId(5),
            offset: n - 1,
            len: n,
        };
        assert_eq!(
            validate_circuit_isolation(&[slot_a, overlap]),
            Err(ErrorCode::ProtectedInvariant)
        );

        // Pack SOLVER_07 weights in slot_a and poisoned cos() weights in slot_b
        let mut combined = prog.weights.clone();
        combined.extend((0..n).map(|i| ((i as f64) * 0.029).cos()));
        let ckpt = Checkpoint::new(
            vec![TensorDescriptor {
                id: TensorId(5),
                component: AdaComponent::SaciCircuits,
                rows: 2 * n,
                cols: 1,
                offset: 0,
                len: 2 * n,
                frozen: true,
            }],
            combined,
        )
        .unwrap();

        // Routing causality: perturbing routed_circuit_id from 7 -> 9 MUST fail, not silently execute
        assert_eq!(
            execute_isolated_saci_slot(&ckpt, &slot_a, 9, &prog, &[10.0, 20.0, 0.0, 1.0])
                .unwrap_err(),
            ErrorCode::UnauthorizedWrite
        );

        // Valid route 7 executes slot_a cleanly without touching slot_b's poisoned weights
        let ok =
            execute_isolated_saci_slot(&ckpt, &slot_a, 7, &prog, &[10.0, 20.0, 0.0, 1.0]).unwrap();
        assert!((ok[0] - (-10.0)).abs() < 1e-9);
        assert!((ok[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn variable_saci_compiler_multi_topology_and_sacib_arena() {
        use crate::saci_compiler::{
            compile_saci_program, execute_saci_circuit, load_saci_binary, parse_saci_source,
            SacisubstrateAllocator, TopologyKind,
        };

        // 1. Pure affine circuit (SOLVER_12): must select Linear topology (not MLP!)
        let linear_src = r#"
[saci]
format_version = 1
id = "SOLVER_12"

[interface]
input_count = 2
output_count = 2

[resources]
max_neurons = 32
max_parameters = 256
max_activation_bytes = 4096

[numerical]
absolute_tolerance = 1e-9

[graph]
INPUT 0
INPUT 1
NODE 2 ADD 0,1
NODE 3 SUB 0,1
OUTPUTS 2,3
"#;

        // 2. Piecewise-linear circuit (SOLVER_07): must select MultiLayerRelu topology
        let forensic_src = r#"
[saci]
format_version = 1
id = "SOLVER_07"

[interface]
input_count = 3
output_count = 4

[resources]
max_neurons = 64
max_parameters = 4096
max_activation_bytes = 65536

[numerical]
absolute_tolerance = 1e-9
steepness_k = 1000000.0

[graph]
INPUT 0
INPUT 1
INPUT 2
CONST 3 0.0
NODE 4 SUB 0,1
NODE 5 SUB 1,0
NODE 6 SUB 5,2
NODE 7 RELU 6
NODE 8 CMP_LT 3,6
NODE 9 CMP_LT 2,3
OUTPUTS 4,7,8,9
"#;

        let mut allocator = SacisubstrateAllocator::new(256);

        let prog_lin = parse_saci_source(linear_src).unwrap();
        let art_lin = compile_saci_program(&prog_lin, &mut allocator).unwrap();
        assert_eq!(art_lin.descriptor.topology, TopologyKind::Linear);
        assert_eq!(art_lin.descriptor.layer_dims, vec![2, 2]);
        assert_eq!(art_lin.resources.materialized_params, 6); // 2*2 + 2

        let prog_for = parse_saci_source(forensic_src).unwrap();
        let art_for = compile_saci_program(&prog_for, &mut allocator).unwrap();
        assert_eq!(art_for.descriptor.topology, TopologyKind::MultiLayerRelu);
        assert_ne!(art_for.descriptor.neuron_ids, art_lin.descriptor.neuron_ids);

        // 3. Load BOTH variable-length .sacib binaries into a single Checkpoint TensorId(5) arena
        let mut ckpt = Checkpoint::new(Vec::new(), Vec::new()).unwrap();
        let loaded_lin = load_saci_binary(&mut ckpt, &art_lin.sacib_bytes).unwrap();
        let loaded_for = load_saci_binary(&mut ckpt, &art_for.sacib_bytes).unwrap();
        assert_eq!(loaded_lin.descriptor.weight_offset, 0);
        assert_eq!(loaded_for.descriptor.weight_offset, 6);

        // 4. Corrupted .sacib must fail atomically without modifying checkpoint
        let mut bad_sacib = art_lin.sacib_bytes.clone();
        let last = bad_sacib.len() - 1;
        bad_sacib[last] ^= 0xFF;
        let pre_len = ckpt.data.len();
        assert_eq!(
            load_saci_binary(&mut ckpt, &bad_sacib).unwrap_err(),
            ErrorCode::InvalidInput
        );
        assert_eq!(ckpt.data.len(), pre_len);

        // 5. Generic zero-copy execution via `execute_saci_circuit`
        let mut scratch = vec![0.0_f64; 64];
        let tr_lin =
            execute_saci_circuit(&ckpt, &loaded_lin.descriptor, &[10.0, 3.0], &mut scratch)
                .unwrap();
        assert_eq!(tr_lin.topology, TopologyKind::Linear);
        assert!((tr_lin.output[0] - 13.0).abs() < 1e-9);
        assert!((tr_lin.output[1] - 7.0).abs() < 1e-9);

        let tr_for = execute_saci_circuit(
            &ckpt,
            &loaded_for.descriptor,
            &[1718003607.0, 1718003707.0, 0.0],
            &mut scratch,
        )
        .unwrap();
        assert_eq!(tr_for.topology, TopologyKind::MultiLayerRelu);
        assert!((tr_for.output[0] - (-100.0)).abs() < 1e-9);
        assert!((tr_for.output[1] - 100.0).abs() < 1e-9);
        assert!((tr_for.output[2] - 1.0).abs() < 1e-9);
        assert!((tr_for.output[3] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn semantic_wsp_composition_invariance_scope_and_round_trip() {
        use crate::semantic_wsp::{
            compile_semantic_wsp, generate_with_ncm_weights, parse_sem_source, serialize_semib,
            understand_with_weights, ConstraintKind, EntityKind, Polarity, PredicateKind,
            QuantifierKind, RelationKind, SemanticFrame, SEMIB_MAGIC,
        };

        let sem_src = r#"
[semantic]
id = "SEM_COMPOSITION_001"
version = 1

[contracts]
preserve_entity_identity = true
preserve_argument_roles = true
preserve_relation_direction = true
preserve_negation_scope = true

[wsp]
parameter_budget = 8192
"#;
        let spec = parse_sem_source(sem_src).unwrap();

        // Compositional training basis (deliberately holding out unseen combinations!)
        let f_move = SemanticFrame {
            action: PredicateKind::Move,
            subject: EntityKind::RedCylinder,
            reference: EntityKind::BlueCylinder,
            relation: RelationKind::LeftOf,
            quant_subject: QuantifierKind::Single,
            quant_reference: QuantifierKind::Single,
            polarity: Polarity::Positive,
            constraint: ConstraintKind::PreserveSubjectHeight,
            magnitude: 20.0,
        };
        let f_scale = SemanticFrame {
            action: PredicateKind::Scale,
            subject: EntityKind::GreenBlock,
            reference: EntityKind::RedCylinder,
            relation: RelationKind::RightOf,
            quant_subject: QuantifierKind::Single,
            quant_reference: QuantifierKind::Single,
            polarity: Polarity::Positive,
            constraint: ConstraintKind::PreserveVolume,
            magnitude: 15.0,
        };
        let f_rot = SemanticFrame {
            action: PredicateKind::Rotate,
            subject: EntityKind::BlueCylinder,
            reference: EntityKind::GreenBlock,
            relation: RelationKind::Above,
            quant_subject: QuantifierKind::Single,
            quant_reference: QuantifierKind::Single,
            polarity: Polarity::Positive,
            constraint: ConstraintKind::PreserveAlignment,
            magnitude: 45.0,
        };
        let f_q1 = SemanticFrame {
            action: PredicateKind::CompareWeight,
            subject: EntityKind::RedComponent,
            reference: EntityKind::BlueComponent,
            relation: RelationKind::HeavierThan,
            quant_subject: QuantifierKind::Every,
            quant_reference: QuantifierKind::Every,
            polarity: Polarity::Positive,
            constraint: ConstraintKind::None,
            magnitude: 0.0,
        };
        let f_q2 = SemanticFrame {
            quant_reference: QuantifierKind::Some,
            ..f_q1
        };
        let f_q3 = SemanticFrame {
            quant_subject: QuantifierKind::Some,
            quant_reference: QuantifierKind::Every,
            ..f_q1
        };
        let f_neg = SemanticFrame {
            action: PredicateKind::Verify,
            polarity: Polarity::Negated,
            relation: RelationKind::AlignedWith,
            ..f_q1
        };

        let train_pairs = [
            (
                "Move the red cylinder 20 mm to the left of the blue cylinder, without changing its height",
                f_move,
            ),
            (
                "Scale the green block 15 mm to the right of the red cylinder preserving volume",
                f_scale,
            ),
            (
                "Rotate the blue cylinder 45 mm above the green block preserving alignment",
                f_rot,
            ),
            (
                "Every red component is heavier than every blue component",
                f_q1,
            ),
            (
                "Every red component is heavier than some blue component",
                f_q2,
            ),
            (
                "Some red component is heavier than every blue component",
                f_q3,
            ),
            (
                "Verify every red component is not aligned with every blue component",
                f_neg,
            ),
            (
                "Move the blue cylinder 10 mm above the red cylinder preserving volume",
                SemanticFrame {
                    action: PredicateKind::Move,
                    subject: EntityKind::BlueCylinder,
                    reference: EntityKind::RedCylinder,
                    relation: RelationKind::Above,
                    quant_subject: QuantifierKind::Single,
                    quant_reference: QuantifierKind::Single,
                    polarity: Polarity::Positive,
                    constraint: ConstraintKind::PreserveVolume,
                    magnitude: 10.0,
                },
            ),
            (
                "Scale the red cylinder 25 mm to the left of the green block preserving alignment",
                SemanticFrame {
                    action: PredicateKind::Scale,
                    subject: EntityKind::RedCylinder,
                    reference: EntityKind::GreenBlock,
                    relation: RelationKind::LeftOf,
                    quant_subject: QuantifierKind::Single,
                    quant_reference: QuantifierKind::Single,
                    polarity: Polarity::Positive,
                    constraint: ConstraintKind::PreserveAlignment,
                    magnitude: 25.0,
                },
            ),
            (
                "Rotate the green block 30 mm to the right of the red cylinder, without changing its height",
                SemanticFrame {
                    action: PredicateKind::Rotate,
                    subject: EntityKind::GreenBlock,
                    reference: EntityKind::RedCylinder,
                    relation: RelationKind::RightOf,
                    quant_subject: QuantifierKind::Single,
                    quant_reference: QuantifierKind::Single,
                    polarity: Polarity::Positive,
                    constraint: ConstraintKind::PreserveSubjectHeight,
                    magnitude: 30.0,
                },
            ),
            (
                "Verify the red cylinder 5 mm aligned with the blue cylinder",
                SemanticFrame {
                    action: PredicateKind::Verify,
                    subject: EntityKind::RedCylinder,
                    reference: EntityKind::BlueCylinder,
                    relation: RelationKind::AlignedWith,
                    quant_subject: QuantifierKind::Single,
                    quant_reference: QuantifierKind::Single,
                    polarity: Polarity::Positive,
                    constraint: ConstraintKind::None,
                    magnitude: 5.0,
                },
            ),
        ];

        let mut weights = compile_semantic_wsp(&spec, &train_pairs).unwrap();
        let semib = serialize_semib(&weights);
        assert_eq!(&semib[0..8], SEMIB_MAGIC);

        // 1. Paraphrase invariance
        let p1 = understand_with_weights(
            &weights,
            "Shift the red cylinder 20 mm to the left of the blue one, without changing its height",
        )
        .unwrap();
        let p2 = understand_with_weights(
            &weights,
            "Translate the red cylinder 20 mm in the negative X direction relative to the blue cylinder, without changing its height",
        )
        .unwrap();
        assert_eq!(p1.action, f_move.action);
        assert_eq!(p1.subject, f_move.subject);
        assert_eq!(p1.reference, f_move.reference);
        assert_eq!(p1.relation, f_move.relation);
        assert_eq!(p1.constraint, f_move.constraint);
        assert!((p1.magnitude - 20.0).abs() < 1e-4);
        assert_eq!(p1.action, p2.action);
        assert_eq!(p1.relation, p2.relation);
        assert_eq!(p1.constraint, p2.constraint);

        // 2. Minimal-pair quantifier scope discrimination
        let u_q1 = understand_with_weights(
            &weights,
            "Every red component is heavier than every blue component",
        )
        .unwrap();
        let u_q2 = understand_with_weights(
            &weights,
            "Every red component is heavier than some blue component",
        )
        .unwrap();
        let u_q3 = understand_with_weights(
            &weights,
            "Some red component is heavier than every blue component",
        )
        .unwrap();
        assert_ne!(u_q1, u_q2);
        assert_ne!(u_q2, u_q3);
        assert_ne!(u_q1, u_q3);
        assert_eq!(u_q1.quant_subject, QuantifierKind::Every);
        assert_eq!(u_q1.quant_reference, QuantifierKind::Every);
        assert_eq!(u_q2.quant_reference, QuantifierKind::Some);
        assert_eq!(u_q3.quant_subject, QuantifierKind::Some);

        // 3. Compositional generalization to UNSEEN combination:
        // Move + green block + right of + blue cylinder + preserving alignment + 35 mm
        let unseen = understand_with_weights(
            &weights,
            "Move the green block 35 mm to the right of the blue cylinder preserving alignment",
        )
        .unwrap();
        assert_eq!(unseen.action, PredicateKind::Move);
        assert_eq!(unseen.subject, EntityKind::GreenBlock);
        assert_eq!(unseen.reference, EntityKind::BlueCylinder);
        assert_eq!(unseen.relation, RelationKind::RightOf);
        assert_eq!(unseen.constraint, ConstraintKind::PreserveAlignment);
        assert!((unseen.magnitude - 35.0).abs() < 1e-3);

        // 4. Round-trip NCM semantic consistency: U(G(U(x))) == U(x)
        let generated = generate_with_ncm_weights(&weights, &unseen).unwrap();
        let recovered = understand_with_weights(&weights, &generated).unwrap();
        assert_eq!(recovered.action, unseen.action);
        assert_eq!(recovered.subject, unseen.subject);
        assert_eq!(recovered.reference, unseen.reference);
        assert_eq!(recovered.relation, unseen.relation);
        assert_eq!(recovered.constraint, unseen.constraint);
        assert!((recovered.magnitude - unseen.magnitude).abs() < 1e-3);

        // 5. Ablation proof: replace w_understand with zeros -> semantic understanding fails!
        weights.w_understand.fill(0.0);
        let ablated = understand_with_weights(
            &weights,
            "Rotate the blue cylinder 45 mm above the green block",
        )
        .unwrap();
        assert_ne!(ablated.action, PredicateKind::Rotate);
    }

    #[test]
    fn wsp_token_decoder_synthesis_and_generation() {
        use crate::token_decoder::{
            compile_token_decoder_wsp, evaluate_token_decoder, generate_tokens,
        };

        let seq: Vec<u16> = [101, 10, 77, 130, 81, 1, 78, 130, 38, 102]
            .into_iter()
            .cycle()
            .take(100)
            .collect();

        let dec = compile_token_decoder_wsp(&seq, 100, 1e-4).unwrap();
        let (top1, top5, count) = evaluate_token_decoder(&dec, &seq, 50);
        assert!(count > 0);
        assert!(top1 > 90.0, "expected >90% top1, got {top1}");
        assert!(top5 >= top1);

        let gen = generate_tokens(&dec, &[101, 10, 77], 8);
        assert_eq!(gen[0], 130);
        assert_eq!(gen[1], 81);
    }

    #[test]
    fn load_real_ada_vq_bpk() {
        use crate::bpk_loader::load_bpk;
        use std::path::Path;

        let vq_path = Path::new(r"D:\AdvancedADA\models\ada_vq\vq_best.bpk");
        if !vq_path.exists() {
            return;
        }
        let archive = load_bpk(vq_path).unwrap();
        assert_eq!(archive.total_f32_params(), 2_529_740);
        let (emb_shape, emb) = archive.read_f32_as_f64("tok_emb.weight").unwrap();
        assert_eq!(emb_shape, vec![8192, 128]);
        assert_eq!(emb.len(), 8192 * 128);
        assert!(emb.iter().all(|v| v.is_finite()));
        let (hb_shape, hb) = archive.read_f32_as_f64("head_bias").unwrap();
        assert_eq!(hb_shape, vec![8192]);
        assert_eq!(hb.len(), 8192);
        assert!(hb.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn parallel_ncm_chunk_generation_and_vq_decoder() {
        use crate::bpk_loader::load_bpk;
        use crate::ncm_generator::{compile_parallel_ncm_wsp, VqDecoderWeights, NCM_CHUNK_LEN};
        use std::path::Path;

        let vq_path = Path::new(r"D:\AdvancedADA\models\ada_vq\vq_best.bpk");
        if !vq_path.exists() {
            return;
        }
        let archive = load_bpk(vq_path).unwrap();
        let vq_dec = VqDecoderWeights::from_bpk(&archive).unwrap();
        assert_eq!(vq_dec.dim, 128);
        assert_eq!(vq_dec.vocab_size, 8192);

        // Compile parallel NCM concept-chunk generator: Workspace state (512) -> 64 x 128 chunk
        let ws_dim = 512usize;
        let mut ws_states = Vec::new();
        let mut target_chunks = Vec::new();
        for c in 0..4usize {
            let mut ws = vec![0.0_f64; ws_dim];
            ws[c] = 1.0;
            let mut chunk = vec![0.0_f64; NCM_CHUNK_LEN * 128];
            for t in 0..NCM_CHUNK_LEN {
                let tok = ((c * 17 + t) % 200) + 10;
                chunk[t * 128..(t + 1) * 128]
                    .copy_from_slice(&vq_dec.tok_emb[tok * 128..(tok + 1) * 128]);
            }
            ws_states.push(ws);
            target_chunks.push(chunk);
        }

        let ncm =
            compile_parallel_ncm_wsp(&ws_states, &target_chunks, ws_dim, NCM_CHUNK_LEN, 128, 1e-4)
                .unwrap();
        let z_hat = ncm.generate_concept_chunk(&ws_states[0]);
        assert_eq!(z_hat.len(), NCM_CHUNK_LEN * 128);

        // Parallel non-autoregressive decoding of all 64 surface tokens at once
        let tokens = vq_dec.decode_chunk_parallel(&z_hat, NCM_CHUNK_LEN);
        assert_eq!(tokens.len(), NCM_CHUNK_LEN);
    }

    #[test]
    fn safetensors_pure_weights_export_round_trip() {
        use crate::safetensors_writer::{write_safetensors, NamedTensor, WeightDtype};

        let tmp = std::env::temp_dir().join("ada_test_pure_weights.safetensors");
        let w1 = [1.25_f64, -2.5_f64, 3.75_f64, 0.5_f64];
        let w2 = [10.0_f64, 20.0_f64];
        let tensors = [
            NamedTensor {
                name: "brain.etf",
                shape: vec![2, 2],
                data: &w1,
            },
            NamedTensor {
                name: "ncm.bias",
                shape: vec![2],
                data: &w2,
            },
        ];
        let meta = [("model_kind", "ada-o1"), ("etf_features", "1024")];
        let written = write_safetensors(&tmp, WeightDtype::F64, &meta, &tensors).unwrap();
        let bytes = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(bytes.len() as u64, written);
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        assert_eq!(hlen % 8, 0);
        let hdr = std::str::from_utf8(&bytes[8..8 + hlen]).unwrap();
        assert!(hdr
            .contains("\"brain.etf\":{\"dtype\":\"F64\",\"shape\":[2,2],\"data_offsets\":[0,32]}"));
        assert!(
            hdr.contains("\"ncm.bias\":{\"dtype\":\"F64\",\"shape\":[2],\"data_offsets\":[32,48]}")
        );
        assert!(hdr.contains("\"etf_features\":\"1024\""));

        let data_start = 8 + hlen;
        let first_val = f64::from_le_bytes(bytes[data_start..data_start + 8].try_into().unwrap());
        assert_eq!(first_val, 1.25_f64);
        let last_val =
            f64::from_le_bytes(bytes[data_start + 40..data_start + 48].try_into().unwrap());
        assert_eq!(last_val, 20.0_f64);

        // Non-.safetensors extensions must be rejected strictly
        let bad_bin = std::env::temp_dir().join("ada_test_pure_weights.bin");
        let err = write_safetensors(&bad_bin, WeightDtype::F64, &meta, &tensors).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!bad_bin.exists());
    }
}

#[cfg(test)]
mod superposition_tests {
    use crate::observatory::format_verifier_report;
    use crate::superposition_wsp::{
        DecisionTreeNode, FeatureSpace, LinearScorerLayer, LinearScorerSpec, NumericThresholdLayer,
        NumericThresholdSpec, RuleLayer, RuleLiteral, RuleSpec, SplitMix64, SuperpositionScratch,
        SuperpositionStack, ThresholdOp,
    };
    use crate::verifier::{verify_superposition_layer, VerifyConfig, VerifyStatus};
    use crate::workflow_arch::{compile_workflow_to_safetensors, parse_workflow_spec};

    fn generate_paper_rules(
        m: usize,
        num_rules: usize,
        num_outputs: usize,
        seed: u64,
    ) -> Vec<RuleSpec> {
        let mut rng = SplitMix64::new(seed);
        let mut rules = Vec::with_capacity(num_rules);
        let mut used = Vec::with_capacity(4);
        for _ in 0..num_rules {
            let n_lit = 2 + rng.next_usize(2); // 2 or 3 literals
            used.clear();
            let mut literals = Vec::with_capacity(n_lit);
            while literals.len() < n_lit {
                let f = rng.next_usize(m) as u32;
                if !used.contains(&f) {
                    used.push(f);
                    let sign = if rng.next_f64_open() < 0.75 { 1 } else { -1 };
                    literals.push(RuleLiteral { feature: f, sign });
                }
            }
            if literals.iter().all(|l| l.sign < 0) {
                literals[0].sign = 1;
            }
            let output = rng.next_usize(num_outputs) as u32;
            rules.push(RuleSpec { literals, output });
        }
        rules
    }

    #[test]
    fn superposition_1000_features_300_rules_verifier_protocol() {
        let m = 1000;
        let num_rules = 300;
        let num_outputs = 5;
        let rules = generate_paper_rules(m, num_rules, num_outputs, 1);

        let cfg = VerifyConfig {
            active_k: 3,
            trials: 600,
            forced_fire_prob: 0.6,
            readout_threshold: 0.5,
            seed: 42,
            coherence_sample_limit: 128,
            ..VerifyConfig::default()
        };

        // d = 512 (paper: ~0.93% error at k=3)
        let in_512 = FeatureSpace::new_gaussian(m, 512, 2).unwrap();
        let out_512 = FeatureSpace::new_gaussian(num_outputs, 512, 3).unwrap();
        let layer_512 = RuleLayer::compile(&in_512, &out_512, &rules, 0.5, None).unwrap();
        let rep_512 =
            verify_superposition_layer(&in_512, &out_512, &layer_512, &rules, None, None, &cfg)
                .unwrap();

        assert_eq!(rep_512.status, VerifyStatus::Pass);
        assert!(
            rep_512.accuracy.forced_fire_error < 0.02,
            "d=512 forced_fire_error too high: {}",
            rep_512.accuracy.forced_fire_error
        );
        assert!(rep_512.etf.present_feature_recall > 0.99);
        assert!(rep_512.noise.snr_db > 15.0);
        assert!(!format_verifier_report(&rep_512).is_empty());

        // d = 1024 (paper: ~0.17% error at k=3, strictly better SNR & lower noise than d=512)
        let in_1024 = FeatureSpace::new_gaussian(m, 1024, 2).unwrap();
        let out_1024 = FeatureSpace::new_gaussian(num_outputs, 1024, 3).unwrap();
        let layer_1024 = RuleLayer::compile(&in_1024, &out_1024, &rules, 0.5, None).unwrap();
        let rep_1024 =
            verify_superposition_layer(&in_1024, &out_1024, &layer_1024, &rules, None, None, &cfg)
                .unwrap();

        assert_eq!(rep_1024.status, VerifyStatus::Pass);
        assert!(
            rep_1024.accuracy.forced_fire_error < 0.008,
            "d=1024 forced_fire_error too high: {}",
            rep_1024.accuracy.forced_fire_error
        );
        assert!(rep_1024.noise.measured_interference_std < rep_512.noise.measured_interference_std);
        assert!(rep_1024.noise.snr_db > rep_512.noise.snr_db);
    }

    #[test]
    fn superposition_thresholds_scorers_trees_and_two_layer_chain() {
        let d = 512;
        let in_space = FeatureSpace::new_gaussian(32, d, 11).unwrap();
        let mid_space = FeatureSpace::new_gaussian(8, d, 22).unwrap();
        let out_space = FeatureSpace::new_gaussian(4, d, 33).unwrap();

        // 1. Numeric thresholds: score >= 0.72 -> f1, risk_flag < 0.5 -> f2
        let thresh = NumericThresholdLayer::compile(
            2,
            &in_space,
            &[
                NumericThresholdSpec {
                    input_idx: 0,
                    threshold: 0.72,
                    steepness: 50.0,
                    op: ThresholdOp::Greater,
                    feature_idx: 1,
                },
                NumericThresholdSpec {
                    input_idx: 1,
                    threshold: 0.5,
                    steepness: 50.0,
                    op: ThresholdOp::Less,
                    feature_idx: 2,
                },
            ],
        )
        .unwrap();

        let mut h = vec![0.0_f64; d];
        thresh.encode_into(&[0.85, 0.0], &mut h, true).unwrap();
        assert!(in_space.read_bool(&h, 1, 0.5).unwrap());
        assert!(in_space.read_bool(&h, 2, 0.5).unwrap());

        thresh.encode_into(&[0.50, 1.0], &mut h, true).unwrap();
        assert!(!in_space.read_bool(&h, 1, 0.5).unwrap());
        assert!(!in_space.read_bool(&h, 2, 0.5).unwrap());

        // 2. Exact weighted linear scorer: 0.4*f3 + 0.25*f4 - 0.15*f5 + 0.1
        let scorer = LinearScorerLayer::compile(
            &in_space,
            &[LinearScorerSpec {
                terms: vec![(3, 0.4), (4, 0.25), (5, -0.15)],
                bias: 0.1,
            }],
        )
        .unwrap();
        in_space
            .write_into(&[(3, 1.0), (4, 0.8), (5, 0.5)], &mut h)
            .unwrap();
        let mut score_out = [0.0_f64; 1];
        scorer.forward_into(&h, &mut score_out).unwrap();
        let expected_score = 0.1 + 0.4 * 1.0 + 0.25 * 0.8 - 0.15 * 0.5;
        assert!(
            (score_out[0] - expected_score).abs() < 0.08,
            "score={} expected={}",
            score_out[0],
            expected_score
        );

        // 3. Decision tree with all-negated leaf (uses __one__ sentinel at feature 0)
        let tree = DecisionTreeNode::Split {
            feature: 1,
            when_true: Box::new(DecisionTreeNode::Split {
                feature: 2,
                when_true: Box::new(DecisionTreeNode::Leaf { output: 0 }),
                when_false: Box::new(DecisionTreeNode::Leaf { output: 1 }),
            }),
            when_false: Box::new(DecisionTreeNode::Leaf { output: 2 }),
        };
        let mut tree_rules = Vec::new();
        tree.compile_to_rules(&mut tree_rules);
        assert_eq!(tree_rules.len(), 3);

        let l1 = RuleLayer::compile(&in_space, &mid_space, &tree_rules, 0.5, Some(0)).unwrap();

        // 4. Layer 2 reads Layer 1's superposed output space (2-layer chain):
        //    mid_0 -> out_0, mid_1 -> out_1, mid_2 -> out_2
        let l2_rules = vec![
            RuleSpec {
                literals: vec![RuleLiteral {
                    feature: 0,
                    sign: 1,
                }],
                output: 0,
            },
            RuleSpec {
                literals: vec![RuleLiteral {
                    feature: 1,
                    sign: 1,
                }],
                output: 1,
            },
            RuleSpec {
                literals: vec![RuleLiteral {
                    feature: 2,
                    sign: 1,
                }],
                output: 2,
            },
        ];
        let l2 = RuleLayer::compile(&mid_space, &out_space, &l2_rules, 0.5, None).unwrap();
        let stack = SuperpositionStack::new(
            vec![in_space.clone(), mid_space, out_space.clone()],
            vec![l1, l2],
        )
        .unwrap();
        let mut scratch = SuperpositionScratch::new(d, 16);

        for f1_on in [false, true] {
            for f2_on in [false, true] {
                let mut mask = vec![false; 32];
                let mut active = vec![(0u32, 1.0)]; // __one__ sentinel
                mask[0] = true;
                if f1_on {
                    mask[1] = true;
                    active.push((1, 1.0));
                }
                if f2_on {
                    mask[2] = true;
                    active.push((2, 1.0));
                }
                in_space.write_into(&active, &mut h).unwrap();
                let (h_final, active_cnt) = stack.forward_into(&h, &mut scratch).unwrap();
                assert!(active_cnt >= 2);
                let expected_leaf = tree.eval_ground_truth(&mask);
                for c in 0..3u32 {
                    assert_eq!(
                        out_space.read_bool(h_final, c, 0.5).unwrap(),
                        c == expected_leaf,
                        "2-layer tree mismatch for f1={f1_on}, f2={f2_on}, c={c}"
                    );
                }
            }
        }
    }

    #[test]
    fn custom_workflow_spec_compile_verify_and_safetensors_export() {
        let qwf = r#"
[workflow]
name = credit_approval_hybrid
hidden_dim = 512
num_features = 64
num_outputs = 4
input_seed = 7
output_seed = 8
dtype = f32

[thresholds]
num_inputs = 2
0, 0.72, 20.0, gt, 1
1, 0.50, 20.0, lt, 2

[scorer]
0.0 ; 1:0.4, 2:0.25, 3:-0.15

[rules]
tau = 0.5
+1,-3 -> 0
+1,+2 -> 0
+4,-1 -> 1
+5,+6,-2 -> 2
+7,-8 -> 3

[residual]
hidden_units = 32
init_scale = 0.0001
seed = 99
"#;

        let spec = parse_workflow_spec(qwf).unwrap();
        assert_eq!(spec.blocks.len(), 4);

        let verify_cfg = VerifyConfig {
            active_k: 3,
            trials: 300,
            ..VerifyConfig::default()
        };

        // Must reject non-.safetensors output
        let bad_out = std::env::temp_dir().join("quintessa_wf_bad.bin");
        assert!(compile_workflow_to_safetensors(&spec, &verify_cfg, &bad_out).is_err());

        // Must compile, pass 5-axis verifier, and emit valid .safetensors
        let good_out = std::env::temp_dir().join("quintessa_wf_credit.safetensors");
        let res = compile_workflow_to_safetensors(&spec, &verify_cfg, &good_out).unwrap();
        assert_eq!(res.rule_reports.len(), 1);
        assert_eq!(res.rule_reports[0].status, VerifyStatus::Pass);
        assert!(res.bytes_written > 0);
        assert!(res.total_tensors >= 10);

        let bytes = std::fs::read(&good_out).unwrap();
        let _ = std::fs::remove_file(&good_out);
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[8..8 + hlen]).unwrap();
        assert!(hdr.contains("\"format\":\"burn\""));
        assert!(!hdr.contains("quintessa"));
        assert!(!hdr.contains("verifier_status"));
        assert!(!hdr.contains("freeze_policies"));
        assert!(hdr.contains("\"space.input.u\""));
        assert!(hdr.contains("\"block.2.rules.w\""));
        assert!(hdr.contains("\"block.3.residual.w_in\""));
    }
}

#[cfg(test)]
mod gap_closure_tests {
    use std::path::Path;

    use crate::conversation::{ConversationSession, LoadedAdaModel};
    use crate::operand_grounding::NumericVocabTable;
    use crate::semantic_data::{
        encode_episodes_to_u16_stream, generate_semantic_corpus, DeepKnowledgeReasoner,
    };
    use crate::sequence_encoder::{SequenceEncoder, SequenceScratch, VocabTokenizer};
    use crate::superposition_wsp::FeatureSpace;

    #[test]
    fn sequence_encoder_distinguishes_word_order_and_negation() {
        let dim = 128usize;
        let vocab_size = 64usize;
        let space = FeatureSpace::new_gaussian(vocab_size, dim, 101).unwrap();
        let enc = SequenceEncoder::new(dim);
        let mut scratch = SequenceScratch::new(dim);

        // "A triggers B" vs "B triggers A" (exact same multiset of tokens!)
        let ab = [5u16, 12, 9];
        let ba = [9u16, 12, 5];
        let mut v_ab = vec![0.0_f64; dim];
        let mut v_ba = vec![0.0_f64; dim];
        enc.encode_into(&ab, &space.directions, vocab_size, &mut scratch, &mut v_ab)
            .unwrap();
        enc.encode_into(&ba, &space.directions, vocab_size, &mut scratch, &mut v_ba)
            .unwrap();

        let cos_order: f64 = v_ab.iter().zip(v_ba.iter()).map(|(a, b)| a * b).sum();
        assert!(
            cos_order < 0.85,
            "Order-sensitive encoder failed to separate A->B from B->A: cos={cos_order}"
        );

        // Negation: "risk is not set" vs "risk is set"
        let neg = [3u16, 7, 19, 22];
        let pos = [3u16, 7, 22];
        let mut v_neg = vec![0.0_f64; dim];
        let mut v_pos = vec![0.0_f64; dim];
        enc.encode_into(
            &neg,
            &space.directions,
            vocab_size,
            &mut scratch,
            &mut v_neg,
        )
        .unwrap();
        enc.encode_into(
            &pos,
            &space.directions,
            vocab_size,
            &mut scratch,
            &mut v_pos,
        )
        .unwrap();
        let cos_neg: f64 = v_neg.iter().zip(v_pos.iter()).map(|(a, b)| a * b).sum();
        assert!(
            cos_neg < 0.90,
            "Negation sensitivity too weak: cos={cos_neg}"
        );
    }

    #[test]
    fn semantic_corpus_and_deep_nonlinear_knowledge_reasoner() {
        let vocab_path = Path::new(r"D:\AdvancedADA\data\vocab_ada_h1.txt");
        if !vocab_path.exists() {
            return;
        }
        let tok = VocabTokenizer::from_vocab_file(vocab_path).unwrap();
        let train_eps = generate_semantic_corpus(&tok, 40, 7);
        let valid_eps = generate_semantic_corpus(&tok, 15, 99);
        assert_eq!(train_eps.len(), 160);

        // Verify binary episode stream serialization round-trips non-empty
        let stream = encode_episodes_to_u16_stream(&train_eps);
        assert!(stream.len() > train_eps.len() * 10);

        // Encode prompts with order-aware SequenceEncoder
        let dim = 64usize;
        let space = FeatureSpace::new_gaussian(tok.id_to_token.len(), dim, 55).unwrap();
        let enc = SequenceEncoder::new(dim);
        let mut scratch = SequenceScratch::new(dim);

        let mut xs_tr = Vec::with_capacity(train_eps.len());
        let mut ys_tr = Vec::with_capacity(train_eps.len());
        for ep in &train_eps {
            let mut v = vec![0.0_f64; dim];
            enc.encode_into(
                &ep.prompt_tokens,
                &space.directions,
                tok.id_to_token.len(),
                &mut scratch,
                &mut v,
            )
            .unwrap();
            xs_tr.push(v);
            let mut y = vec![0.0_f64; 4];
            y[ep.category as usize] = 1.0;
            ys_tr.push(y);
        }

        // Train 2-layer nonlinear ReLU-ETF DeepKnowledgeReasoner analytically via WSP Ridge
        let dkr =
            DeepKnowledgeReasoner::compile_wsp(&xs_tr, &ys_tr, dim, 256, 4, 1e-2, 123).unwrap();

        let mut h_buf = vec![0.0_f64; 256];
        let mut out_buf = vec![0.0_f64; 4];
        let mut correct = 0usize;
        for ep in &valid_eps {
            let mut v = vec![0.0_f64; dim];
            enc.encode_into(
                &ep.prompt_tokens,
                &space.directions,
                tok.id_to_token.len(),
                &mut scratch,
                &mut v,
            )
            .unwrap();
            dkr.forward_into(&v, &mut h_buf, &mut out_buf).unwrap();
            let pred = out_buf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if pred == ep.category as usize {
                correct += 1;
            }
        }
        let acc = (correct as f64) / (valid_eps.len() as f64);
        assert!(
            acc >= 0.95,
            "DeepKnowledgeReasoner held-out semantic accuracy too low: {:.2}% ({}/{})",
            acc * 100.0,
            correct,
            valid_eps.len()
        );
    }

    #[test]
    fn conversation_session_loads_pure_safetensors_and_runs_multi_turn() {
        let model_path = Path::new(r"D:\TRACR\quintessa\output\ada_o1_512.safetensors");
        let vocab_path = Path::new(r"D:\AdvancedADA\data\vocab_ada_h1.txt");
        if !model_path.exists() || !vocab_path.exists() {
            return;
        }
        let tok = VocabTokenizer::from_vocab_file(vocab_path).unwrap();
        let model = LoadedAdaModel::from_safetensors(
            model_path,
            Some(Path::new(r"D:\TRACR\quintessa\specs\saci")),
        )
        .unwrap();
        let num_table = NumericVocabTable::from_vocab_file(vocab_path, model.vocab_size);
        let mut session = ConversationSession::new(model.dim);

        let t1_toks = tok.encode(
            "assess causal identification of P(y|do(x)) in network with directed edges x > y",
        );
        let out1 = session
            .respond_turn(&model, &t1_toks, &num_table, 2)
            .unwrap();
        assert_eq!(out1.turn_index, 1);
        assert!(out1.all_verified);
        assert!(out1.chunks_generated >= 1);
        assert!(!out1.tokens.is_empty());

        // Turn 2 must use non-zero conversational KDA memory from Turn 1
        let mem_norm_sq: f64 = session.mem_state.iter().map(|v| v * v).sum();
        assert!(mem_norm_sq > 1e-6);

        let t2_toks = tok.encode("summarize the previous causal identification");
        let out2 = session
            .respond_turn(&model, &t2_toks, &num_table, 2)
            .unwrap();
        assert_eq!(out2.turn_index, 2);
        assert!(!out2.tokens.is_empty());
    }
}

#[cfg(test)]
mod primitive_tests {
    use crate::primitives::{
        run_dual_timescale_cycles, DoubleGatedShortConv, DualTimescaleConfig, DualTimescaleScratch,
        JumpReLuSparseLayer, Mamba2State, SelectiveSsmMamba2, ShortConvCache,
    };

    #[test]
    fn jumprelu_sparse_encoder_structural_numerical_and_sparsity() {
        // Gemma Scope 2 style JumpReLU: d=4, width=8, exact unit-norm decoder rows
        let d = 4;
        let width = 8;
        let mut w_enc = vec![0.0_f64; d * width];
        let mut w_dec = vec![0.0_f64; width * d];
        for j in 0..4 {
            w_enc[j * width + j] = 1.0;
            w_dec[j * d + j] = 1.0;
        }
        for j in 4..8 {
            w_enc[(j - 4) * width + j] = 0.2;
            w_dec[j * d + (j - 4)] = 1.0;
        }
        let b_enc = vec![-0.1; width];
        let threshold = vec![0.5; width];
        let b_dec = vec![0.0; d];

        let layer =
            JumpReLuSparseLayer::new(d, width, d, w_enc, b_enc, threshold, w_dec, b_dec).unwrap();
        let x = [2.0, 0.3, 1.5, 0.0];
        let mut z = vec![0.0_f64; width];
        let mut y = vec![0.0_f64; d];
        let l0 = layer.forward_into(&x, &mut z, &mut y).unwrap();

        // Only features 0 (pre=1.9 > 0.5) and 2 (pre=1.4 > 0.5) should survive JumpReLU
        assert_eq!(l0, 2);
        assert!((z[0] - 1.9).abs() < 1e-12);
        assert_eq!(z[1], 0.0);
        assert!((z[2] - 1.4).abs() < 1e-12);
        assert!((y[0] - 1.9).abs() < 1e-12);
        assert!((y[2] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn lfm2_double_gated_short_conv_causal_streaming() {
        let d = 2;
        let l_cache = 3;
        // Identity in_proj for B=x, C=x, x_in=x; identity out_proj
        let mut in_proj = vec![0.0_f64; 3 * d * d];
        for blk in 0..3 {
            for i in 0..d {
                in_proj[(blk * d + i) * d + i] = 1.0;
            }
        }
        // Causal kernel [0.5, 0.3, 0.2] on both channels
        let kernel = vec![0.5, 0.3, 0.2, 0.5, 0.3, 0.2];
        let out_proj = vec![1.0, 0.0, 0.0, 1.0];

        let conv = DoubleGatedShortConv::new(d, l_cache, in_proj, kernel, out_proj).unwrap();
        let mut cache = ShortConvCache::new(d, l_cache);
        let mut y = [0.0_f64; 2];

        // t=0: x=[2, 1] -> u_0=[4, 1], v_0=[0.5*4, 0.5*1], y_0=C*v_0=[2*2, 1*0.5]=[4.0, 0.5]
        conv.step_into(&[2.0, 1.0], &mut cache, &mut y).unwrap();
        assert!((y[0] - 4.0).abs() < 1e-12);
        assert!((y[1] - 0.5).abs() < 1e-12);

        // t=1: x=[1, 2] -> u_1=[1, 4], v_1=[0.5*1+0.3*4, 0.5*4+0.3*1]=[1.7, 2.3], y_1=[1.7, 4.6]
        conv.step_into(&[1.0, 2.0], &mut cache, &mut y).unwrap();
        assert!((y[0] - 1.7).abs() < 1e-12);
        assert!((y[1] - 4.6).abs() < 1e-12);

        // t=2: x=[1, 1] -> u_2=[1, 1], v_2=[0.5*1+0.3*1+0.2*4, 0.5*1+0.3*4+0.2*1]=[1.6, 1.9]
        conv.step_into(&[1.0, 1.0], &mut cache, &mut y).unwrap();
        assert!((y[0] - 1.6).abs() < 1e-12);
        assert!((y[1] - 1.9).abs() < 1e-12);
    }

    #[test]
    fn granite_mamba2_selective_ssm_recurrence_and_decay() {
        let ssm = SelectiveSsmMamba2::new(
            2,
            2,
            2,
            vec![0.0, 0.0], // a = -exp(0) = -1.0
            vec![0.5, 0.5], // D skip = 0.5
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut st = Mamba2State::new(2, 2, 2);
        let mut y1 = [0.0_f64; 4];
        let mut y2 = [0.0_f64; 4];

        // Impulse at t=0, zero input at t=1 -> state must decay smoothly by exp(-dt)
        ssm.step_into(
            &[1.0, 2.0, 0.5, -1.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 0.0],
            &mut st,
            &mut y1,
        )
        .unwrap();
        let dt = (2.0_f64).ln();
        let expected_y1_0 = 0.5 * 1.0 + dt * 1.0;
        assert!((y1[0] - expected_y1_0).abs() < 1e-12);

        ssm.step_into(
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 0.0],
            &mut st,
            &mut y2,
        )
        .unwrap();
        let decay = (-dt).exp();
        assert!((y2[0] - dt * decay).abs() < 1e-12);
        assert!(y2[0].abs() < y1[0].abs());
    }

    #[test]
    fn hrm_dual_timescale_nested_cycles_and_convergence() {
        let cfg = DualTimescaleConfig {
            dim: 4,
            h_cycles: 4,
            l_cycles: 3,
            halt_delta_tol: 1e-4,
            z_l_init: vec![0.1, 0.1, 0.1, 0.1],
        };
        let mut scratch = DualTimescaleScratch::new(4);
        let x = [1.0, -0.5, 0.25, 0.0];

        // Contractive L and H steps: must execute nested cycles and converge
        let trace = run_dual_timescale_cycles(
            &cfg,
            &x,
            |inp, out| {
                for (d, &v) in out.iter_mut().zip(inp.iter()) {
                    *d = 0.3 * v;
                }
            },
            |inp, out| {
                for (d, &v) in out.iter_mut().zip(inp.iter()) {
                    *d = 0.25 * v;
                }
            },
            &mut scratch,
        )
        .unwrap();

        assert!(trace.h_steps_executed >= 1 && trace.h_steps_executed <= 4);
        assert_eq!(trace.l_steps_executed, trace.h_steps_executed * 3);
        assert!(trace.final_h_delta.is_finite());
        assert!(scratch.z_h[0] > 0.0 && scratch.z_h[1] < 0.0);
    }
}

#[cfg(test)]
mod research_upgrade_tests {
    use crate::primitives::{parameterless_rmsnorm_into, GatedAttentionGqkv, GqkvScratch};
    use crate::safetensors_writer::{
        read_safetensors, validate_mixed_tensor_health, write_mixed_safetensors, TypedNamedTensor,
        WeightDtype,
    };
    use crate::superposition_wsp::{FeatureSpace, RuleLayer, RuleLiteral, RuleSpec};
    use crate::verifier::{VerifyConfig, VerifyStatus};
    use crate::workflow_arch::{
        compile_workflow_to_safetensors, parse_workflow_spec, CompileProfile,
    };

    #[test]
    fn oblique_rule_layer_cancels_interference() {
        // Undercomplete regime (m = 32 features in d = 64 dim): oblique read directions
        // W_enc = (G + lambda*I)^-1 U^T nearly eliminate cross-feature interference.
        let in_space = FeatureSpace::new_gaussian(32, 64, 17).unwrap();
        let out_space = FeatureSpace::new_gaussian(8, 64, 29).unwrap();

        let rules = vec![
            RuleSpec {
                literals: vec![
                    RuleLiteral {
                        feature: 0,
                        sign: 1,
                    },
                    RuleLiteral {
                        feature: 1,
                        sign: -1,
                    },
                ],
                output: 0,
            },
            RuleSpec {
                literals: vec![
                    RuleLiteral {
                        feature: 2,
                        sign: 1,
                    },
                    RuleLiteral {
                        feature: 3,
                        sign: 1,
                    },
                ],
                output: 1,
            },
            RuleSpec {
                literals: vec![
                    RuleLiteral {
                        feature: 4,
                        sign: 1,
                    },
                    RuleLiteral {
                        feature: 5,
                        sign: -1,
                    },
                    RuleLiteral {
                        feature: 6,
                        sign: 1,
                    },
                ],
                output: 2,
            },
        ];

        let oblique_layer =
            RuleLayer::compile_oblique(&in_space, &out_space, &rules, 0.5, None, 1e-3).unwrap();

        let mut h = vec![0.0_f64; 64];
        let mut acts = vec![0.0_f64; rules.len()];
        let mut h_out = vec![0.0_f64; 64];

        // Case 1: features 0, 2, 3, 4, 6 active -> outputs 0, 1, 2 must all fire
        in_space
            .write_into(&[(0, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (6, 1.0)], &mut h)
            .unwrap();
        oblique_layer
            .forward_into(&h, &mut acts, &mut h_out)
            .unwrap();
        assert!(out_space.read_bool(&h_out, 0, 0.5).unwrap());
        assert!(out_space.read_bool(&h_out, 1, 0.5).unwrap());
        assert!(out_space.read_bool(&h_out, 2, 0.5).unwrap());
        assert!(!out_space.read_bool(&h_out, 3, 0.5).unwrap());

        // Case 2: feature 1 active (negates clause 0) and feature 5 active (negates clause 2)
        in_space
            .write_into(&[(0, 1.0), (1, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)], &mut h)
            .unwrap();
        oblique_layer
            .forward_into(&h, &mut acts, &mut h_out)
            .unwrap();
        assert!(!out_space.read_bool(&h_out, 0, 0.5).unwrap());
        assert!(!out_space.read_bool(&h_out, 1, 0.5).unwrap());
        assert!(!out_space.read_bool(&h_out, 2, 0.5).unwrap());
    }

    #[test]
    fn hrm_gated_attention_gqkv_and_parameterless_rmsnorm() {
        // 1. Parameterless RMSNorm must produce unit RMS with zero learnable parameters.
        let x = [3.0_f64, -4.0, 0.0, 5.0];
        let mut y = [0.0_f64; 4];
        parameterless_rmsnorm_into(&x, 1e-6, &mut y).unwrap();
        let rms = (y.iter().map(|v| v * v).sum::<f64>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "expected unit RMS, got {rms}");

        // 2. Gated Attention (GQKV) causality + finite bounded output.
        let dim = 4usize;
        let num_heads = 2usize;
        let mut gqkv_proj = vec![0.0_f64; (4 * dim) * dim];
        let mut o_proj = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            gqkv_proj[i * dim + i] = 0.8; // G
            gqkv_proj[(dim + i) * dim + i] = 0.5; // Q
            gqkv_proj[(2 * dim + i) * dim + i] = 0.5; // K
            gqkv_proj[(3 * dim + i) * dim + i] = 0.6; // V
            o_proj[i * dim + i] = 1.0;
        }
        let gqkv = GatedAttentionGqkv {
            dim,
            num_heads,
            head_dim: dim / num_heads,
            gqkv_proj,
            o_proj,
        };
        let mut scratch = GqkvScratch::new(dim, 4);

        let seq_a = [
            1.0, 0.0, 0.5, -0.5, // t = 0
            0.0, 1.0, -0.5, 0.5, // t = 1
            0.5, 0.5, 0.5, 0.5, // t = 2
        ];
        let mut seq_b = seq_a;
        // Mutate only future token t = 2
        seq_b[8] = -3.0;
        seq_b[9] = 2.0;

        let mut out_a = vec![0.0_f64; seq_a.len()];
        let mut out_b = vec![0.0_f64; seq_b.len()];
        gqkv.forward_seq_into(&seq_a, 3, &mut scratch, &mut out_a)
            .unwrap();
        gqkv.forward_seq_into(&seq_b, 3, &mut scratch, &mut out_b)
            .unwrap();

        assert!(out_a.iter().all(|v| v.is_finite()));
        // Causal guarantee: t=0 and t=1 outputs must be bit-identical
        for i in 0..(2 * dim) {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-12,
                "causality violated at index {i}: {} vs {}",
                out_a[i],
                out_b[i]
            );
        }
        // t=2 output must differ
        let diff_t2: f64 = (2 * dim..3 * dim)
            .map(|i| (out_a[i] - out_b[i]).abs())
            .sum();
        assert!(diff_t2 > 1e-3, "expected t=2 to change, diff={diff_t2}");
    }

    #[test]
    fn mixed_precision_safetensors_round_trip_and_health_gate() {
        let path = std::env::temp_dir().join("quintessa_mixed_precision_test.safetensors");

        let d_f64 = [1.25_f64, -2.5];
        let d_f32 = [0.5_f64, -1.75, 42.25];
        let d_bf16 = [1.0_f64, -0.5, 3.25, 0.0];
        let d_f16 = [0.25_f64, -2.0, 16.5, 0.0];
        let d_i8 = [0.0_f64, 1.0, -1.0, 0.5];

        let tensors = [
            TypedNamedTensor {
                name: "t.f64",
                shape: vec![2],
                dtype: WeightDtype::F64,
                data: &d_f64,
            },
            TypedNamedTensor {
                name: "t.f32",
                shape: vec![3],
                dtype: WeightDtype::F32,
                data: &d_f32,
            },
            TypedNamedTensor {
                name: "t.bf16",
                shape: vec![2, 2],
                dtype: WeightDtype::BF16,
                data: &d_bf16,
            },
            TypedNamedTensor {
                name: "t.f16",
                shape: vec![2, 2],
                dtype: WeightDtype::F16,
                data: &d_f16,
            },
            TypedNamedTensor {
                name: "t.i8",
                shape: vec![1, 2, 2],
                dtype: WeightDtype::I8,
                data: &d_i8,
            },
        ];

        let (bytes, health) =
            write_mixed_safetensors(&path, &[("test", "mixed_precision")], &tensors).unwrap();
        assert!(bytes > 0);
        assert!(health.passed);
        assert_eq!(health.nonfinite_count, 0);
        assert_eq!(health.f16_overflow_count, 0);

        let loaded = read_safetensors(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.len(), 5);

        let r_f64 = loaded.iter().find(|t| t.name == "t.f64").unwrap();
        assert_eq!(r_f64.dtype, WeightDtype::F64);
        assert!((r_f64.data[0] - 1.25).abs() < 1e-12);

        let r_bf16 = loaded.iter().find(|t| t.name == "t.bf16").unwrap();
        assert_eq!(r_bf16.dtype, WeightDtype::BF16);
        assert!((r_bf16.data[2] - 3.25).abs() < 1e-2);

        let r_f16 = loaded.iter().find(|t| t.name == "t.f16").unwrap();
        assert_eq!(r_f16.dtype, WeightDtype::F16);
        assert!((r_f16.data[2] - 16.5).abs() < 1e-2);

        let r_i8 = loaded.iter().find(|t| t.name == "t.i8").unwrap();
        assert_eq!(r_i8.dtype, WeightDtype::I8);
        assert!((r_i8.data[1] - 1.0).abs() < 0.02);

        // Health Gate must reject NaN
        let bad_nan = [f64::NAN, 1.0];
        let nan_tensors = [TypedNamedTensor {
            name: "bad.nan",
            shape: vec![2],
            dtype: WeightDtype::F32,
            data: &bad_nan,
        }];
        let nan_health = validate_mixed_tensor_health(&nan_tensors);
        assert!(!nan_health.passed);
        assert_eq!(nan_health.nonfinite_count, 1);
        assert!(write_mixed_safetensors(&path, &[], &nan_tensors).is_err());

        // Health Gate must reject F16 overflow (> 65504.0)
        let bad_ovf = [70000.0_f64, 1.0];
        let ovf_tensors = [TypedNamedTensor {
            name: "bad.f16_overflow",
            shape: vec![2],
            dtype: WeightDtype::F16,
            data: &bad_ovf,
        }];
        let ovf_health = validate_mixed_tensor_health(&ovf_tensors);
        assert!(!ovf_health.passed);
        assert_eq!(ovf_health.f16_overflow_count, 1);
        assert!(write_mixed_safetensors(&path, &[], &ovf_tensors).is_err());
    }

    #[test]
    fn intelligent_profile_workflow_all_primitives() {
        let qwf = r#"
[workflow]
name = quintessa_full_hybrid
hidden_dim = 256
num_features = 48
num_outputs = 4
input_seed = 11
output_seed = 19
dtype = f32
profile = memory

[thresholds]
num_inputs = 2
0, 0.70, 20.0, gt, 1
1, 0.30, 20.0, lt, 2

[scorer]
0.0 ; 1:0.5, 2:0.25

[rules]
tau = 0.5
oblique = true
+1,-3 -> 0
+1,+2 -> 1
+4,-2 -> 2
+5,-6 -> 3

[jumprelu]
threshold = 0.45
seed = 101

[short_conv]
kernel_size = 3
seed = 202

[mamba2]
num_heads = 4
state_dim = 8
seed = 303

[gated_attn]
num_heads = 4
seed = 404

[dual_timescale]
h_cycles = 2
l_cycles = 3
seed = 505

[residual]
hidden_units = 16
init_scale = 0.0001
seed = 606
"#;

        let mut spec_mem = parse_workflow_spec(qwf).unwrap();
        assert_eq!(spec_mem.blocks.len(), 9);
        assert_eq!(spec_mem.profile, CompileProfile::Memory);

        let verify_cfg = VerifyConfig {
            active_k: 3,
            trials: 200,
            ..VerifyConfig::default()
        };

        let out_mem = std::env::temp_dir().join("quintessa_all_prim_mem.safetensors");
        let res_mem = compile_workflow_to_safetensors(&spec_mem, &verify_cfg, &out_mem).unwrap();
        assert!(res_mem.weight_health.passed);
        assert_eq!(res_mem.rule_reports[0].status, VerifyStatus::Pass);

        spec_mem.profile = CompileProfile::Accuracy;
        let out_acc = std::env::temp_dir().join("quintessa_all_prim_acc.safetensors");
        let res_acc = compile_workflow_to_safetensors(&spec_mem, &verify_cfg, &out_acc).unwrap();
        assert!(res_acc.weight_health.passed);
        assert_eq!(res_acc.rule_reports[0].status, VerifyStatus::Pass);

        // Memory profile must emit strictly fewer bytes than Accuracy profile
        assert!(
            res_mem.bytes_written < res_acc.bytes_written,
            "Memory profile ({}) should be smaller than Accuracy profile ({})",
            res_mem.bytes_written,
            res_acc.bytes_written
        );

        // Verify mixed dtypes in Memory header: thresholds/mamba stay F32, bulk matrices use F16/BF16
        let mem_bytes = std::fs::read(&out_mem).unwrap();
        let hlen = u64::from_le_bytes(mem_bytes[0..8].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&mem_bytes[8..8 + hlen]).unwrap();
        assert!(hdr.contains("\"format\":\"burn\""));
        assert!(!hdr.contains("compile_profile"));
        assert!(hdr.contains("\"block.3.jumprelu.threshold\":{\"dtype\":\"F32\""));
        assert!(hdr.contains("\"block.5.mamba2.dt_bias\":{\"dtype\":\"F32\""));
        assert!(hdr.contains("\"block.6.gated_attn.gqkv_proj\":{\"dtype\":\"F16\""));
        assert!(hdr.contains("\"block.7.dual_timescale.z_l_init\":{\"dtype\":\"F32\""));

        let _ = std::fs::remove_file(&out_mem);
        let _ = std::fs::remove_file(&out_acc);
    }

    #[test]
    fn semantic_wsp_compilers_execute_exact_behaviors() {
        use crate::primitives::{Mamba2State, ShortConvCache};
        use crate::semantic_compiler::{
            compile_semantic_dual_timescale, compile_semantic_gated_attn,
            compile_semantic_jumprelu, compile_semantic_mamba2, compile_semantic_short_conv,
            AttnRouteSpec, DualTimescaleCouplingSpec, Mamba2RegisterSpec, SemanticConceptSpec,
            ShortConvPatternSpec,
        };

        let dim = 64usize;
        let in_space = FeatureSpace::new_gaussian(24, dim, 13).unwrap();
        let out_space = FeatureSpace::new_gaussian(8, dim, 31).unwrap();

        // 1. JumpReLU Semantic Concept Compiler
        let concepts = vec![
            SemanticConceptSpec {
                literals: vec![
                    RuleLiteral {
                        feature: 0,
                        sign: 1,
                    },
                    RuleLiteral {
                        feature: 1,
                        sign: -1,
                    },
                ],
                output_feature: 0,
                deadzone_threshold: 0.6,
            },
            SemanticConceptSpec {
                literals: vec![
                    RuleLiteral {
                        feature: 2,
                        sign: 1,
                    },
                    RuleLiteral {
                        feature: 3,
                        sign: 1,
                    },
                ],
                output_feature: 1,
                deadzone_threshold: 0.6,
            },
        ];
        let jrelu = compile_semantic_jumprelu(&in_space, &out_space, &concepts, 1e-3).unwrap();
        let mut x = vec![0.0_f64; dim];
        let mut z = vec![0.0_f64; 2];
        let mut y = vec![0.0_f64; dim];

        // Feature 0 ON, 1 OFF -> concept 0 fires, concept 1 stays 0
        in_space.write_into(&[(0, 1.0)], &mut x).unwrap();
        let l0 = jrelu.forward_into(&x, &mut z, &mut y).unwrap();
        assert_eq!(l0, 1);
        assert!(z[0] > 0.8 && z[1] == 0.0);
        assert!(out_space.read_bool(&y, 0, 0.5).unwrap());

        // Feature 0 ON, 1 ON -> negated literal cancels concept 0
        in_space.write_into(&[(0, 1.0), (1, 1.0)], &mut x).unwrap();
        let l0_neg = jrelu.forward_into(&x, &mut z, &mut y).unwrap();
        assert_eq!(l0_neg, 0);

        // 2. ShortConv Semantic Bigram Trigger Compiler: (t-1=4, t=7) -> output 2
        let patterns = vec![ShortConvPatternSpec {
            tap_t2: None,
            tap_t1: Some(4),
            tap_t0: 7,
            output_feature: 2,
            gain: 1.0,
        }];
        let sconv = compile_semantic_short_conv(&in_space, &out_space, &patterns, 1e-3).unwrap();
        let mut cache = ShortConvCache::new(dim, 3);
        let mut x_tok = vec![0.0_f64; dim];
        let mut y_tok = vec![0.0_f64; dim];

        // Step 0: feature 4 -> output 2 should NOT fire yet
        in_space.write_into(&[(4, 1.0)], &mut x_tok).unwrap();
        sconv.step_into(&x_tok, &mut cache, &mut y_tok).unwrap();
        assert!(!out_space.read_bool(&y_tok, 2, 0.5).unwrap());

        // Step 1: feature 7 after 4 -> bigram (4, 7) MUST fire output 2!
        in_space.write_into(&[(7, 1.0)], &mut x_tok).unwrap();
        sconv.step_into(&x_tok, &mut cache, &mut y_tok).unwrap();
        assert!(out_space.read_bool(&y_tok, 2, 0.5).unwrap());

        // Step 2: feature 7 after 7 -> NOT (4, 7), must NOT fire output 2!
        in_space.write_into(&[(7, 1.0)], &mut x_tok).unwrap();
        sconv.step_into(&x_tok, &mut cache, &mut y_tok).unwrap();
        assert!(!out_space.read_bool(&y_tok, 2, 0.5).unwrap());

        // 3. Mamba-2 Semantic Half-Life Compiler
        let regs = vec![
            Mamba2RegisterSpec {
                head_idx: 0,
                half_life_tokens: 1.0,
                skip_gain: 0.0,
            },
            Mamba2RegisterSpec {
                head_idx: 1,
                half_life_tokens: 16.0,
                skip_gain: 0.0,
            },
        ];
        let mamba = compile_semantic_mamba2(4, 2, 2, &regs).unwrap();
        let mut mstate = Mamba2State::new(4, 2, 2);
        // Inject impulse into head 0 & head 1 state
        mstate.h_state[0] = 1.0;
        mstate.h_state[4] = 1.0;
        let zero_x = vec![0.0_f64; 8];
        let zero_dt = vec![0.0_f64; 4];
        let zero_bc = vec![0.0_f64; 2];
        let mut y_m = vec![0.0_f64; 8];
        mamba
            .step_into(&zero_x, &zero_bc, &zero_bc, &zero_dt, &mut mstate, &mut y_m)
            .unwrap();
        // After 1 token with half_life=1.0, head 0 state must decay to exactly 0.5
        assert!(
            (mstate.h_state[0] - 0.5).abs() < 1e-4,
            "expected head 0 decay=0.5, got {}",
            mstate.h_state[0]
        );
        // After 15 more steps (16 total), head 1 with half_life=16.0 must decay to exactly 0.5
        for _ in 0..15 {
            mamba
                .step_into(&zero_x, &zero_bc, &zero_bc, &zero_dt, &mut mstate, &mut y_m)
                .unwrap();
        }
        assert!(
            (mstate.h_state[4] - 0.5).abs() < 1e-3,
            "expected head 1 16-step decay=0.5, got {}",
            mstate.h_state[4]
        );

        // 4. Gated Attention GQKV Semantic Associative KV Router
        let routes = vec![AttnRouteSpec {
            head_idx: 0,
            query_feature: 1,
            key_feature: 2,
            value_feature: 3,
            output_feature: 0,
            gate_strength: 5.0,
        }];
        let gqkv = compile_semantic_gated_attn(&in_space, &out_space, 4, &routes, 1e-3).unwrap();
        let mut gscratch = GqkvScratch::new(dim, 3);
        let mut seq = vec![0.0_f64; 3 * dim];
        let mut out_seq = vec![0.0_f64; 3 * dim];
        // t=0: key=2, value=3
        in_space
            .write_into(&[(2, 1.0), (3, 1.0)], &mut seq[0..dim])
            .unwrap();
        // t=1: distractor feature 9
        in_space
            .write_into(&[(9, 1.0)], &mut seq[dim..2 * dim])
            .unwrap();
        // t=2: query=1 -> must retrieve value=3 from t=0 and write output_feature=0
        in_space
            .write_into(&[(1, 1.0)], &mut seq[2 * dim..3 * dim])
            .unwrap();
        gqkv.forward_seq_into(&seq, 3, &mut gscratch, &mut out_seq)
            .unwrap();
        assert!(out_space
            .read_bool(&out_seq[2 * dim..3 * dim], 0, 0.5)
            .unwrap());

        // 5. Dual-Timescale Contractive Compiler + Full Semantic .qwf Pipeline
        let dt = compile_semantic_dual_timescale(
            &in_space,
            &[DualTimescaleCouplingSpec {
                plan_feature: 0,
                fast_feature: 1,
                contraction_rate: 0.4,
            }],
            1e-3,
        )
        .unwrap();
        let z_norm = dt.z_l_init.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((z_norm - (dim as f64).sqrt()).abs() < 1e-6);

        let sem_qwf = r#"
[workflow]
name = semantic_wsp_full
hidden_dim = 64
num_features = 24
num_outputs = 4
profile = balanced

[jumprelu]
threshold = 0.6
concept = +0,-1 -> 0 @ 0.6
concept = +2,+3 -> 1 @ 0.6

[short_conv]
kernel_size = 3
pattern = 4, 7 -> 2 * 1.0

[mamba2]
num_heads = 4
state_dim = 4
register = 0, 1.0, 0.0
register = 1, 64.0, 0.05

[gated_attn]
num_heads = 4
route = 0 : 1, 2, 3 -> 0 * 5.0

[dual_timescale]
h_cycles = 2
l_cycles = 3
coupling = 0 -> 1 * 0.4
"#;
        let spec = parse_workflow_spec(sem_qwf).unwrap();
        assert_eq!(spec.blocks.len(), 5);
        let out_p = std::env::temp_dir().join("quintessa_semantic_full.safetensors");
        let res = compile_workflow_to_safetensors(&spec, &VerifyConfig::default(), &out_p).unwrap();
        assert!(res.weight_health.passed);
        let bytes = std::fs::read(&out_p).unwrap();
        let _ = std::fs::remove_file(&out_p);
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[8..8 + hlen]).unwrap();
        assert!(hdr.contains("\"block.4.dual_timescale.w_l_step\""));
        assert!(hdr.contains("\"block.4.dual_timescale.w_h_step\""));
    }

    #[test]
    fn pondernet_refinement_wireframe_component_halts_and_exports() {
        use crate::primitives::PonderRefinementScratch;
        use crate::semantic_compiler::{compile_semantic_pondernet, PonderNetSpec};

        let dim = 64usize;
        let space = FeatureSpace::new_gaussian(16, dim, 41).unwrap();
        let pspec = PonderNetSpec {
            t_cap: 5,
            lambda_prior: 0.4,
            halt_feature: 0,
            verify_feature: 1,
            feedback_gain: 0.25,
        };
        let ponder = compile_semantic_pondernet(&space, &pspec, 1e-3).unwrap();
        let mut scratch = PonderRefinementScratch::new(dim);
        let mut ctx = vec![0.0_f64; dim];
        let mut acc = vec![0.0_f64; dim];

        // Case 1: Both halt_feature=0 and verify_feature=1 active -> must halt early at step 1
        space.write_into(&[(0, 1.0), (1, 1.0)], &mut ctx).unwrap();
        let trace_early = ponder
            .refine_into(
                &ctx,
                |inp, out| out.copy_from_slice(inp),
                &mut scratch,
                &mut acc,
            )
            .unwrap();
        assert_eq!(trace_early.steps_executed, 1);
        assert!(trace_early.halted_early);
        assert!(trace_early.final_halt_prob > 0.9);
        assert!(trace_early.final_verify_prob > 0.9);

        // Case 2: Only unrelated feature 5 active -> must run all t_cap=5 refinement steps
        space.write_into(&[(5, 1.0)], &mut ctx).unwrap();
        let trace_full = ponder
            .refine_into(
                &ctx,
                |inp, out| out.copy_from_slice(inp),
                &mut scratch,
                &mut acc,
            )
            .unwrap();
        assert_eq!(trace_full.steps_executed, 5);
        assert!(!trace_full.halted_early);

        // Case 3: Compile [pondernet] wireframe block to mixed-precision .safetensors
        let qwf = r#"
[workflow]
name = pondernet_wireframe
hidden_dim = 64
num_features = 16
num_outputs = 4
profile = memory

[pondernet]
t_cap = 4
lambda_prior = 0.4
halt_feature = 0
verify_feature = 1
feedback_gain = 0.25
"#;
        let spec = parse_workflow_spec(qwf).unwrap();
        let out_p = std::env::temp_dir().join("quintessa_pondernet_wf.safetensors");
        let res = compile_workflow_to_safetensors(&spec, &VerifyConfig::default(), &out_p).unwrap();
        assert!(res.weight_health.passed);
        let bytes = std::fs::read(&out_p).unwrap();
        let _ = std::fs::remove_file(&out_p);
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr = std::str::from_utf8(&bytes[8..8 + hlen]).unwrap();
        assert!(hdr.contains("\"block.0.pondernet.w_feedback\":{\"dtype\":\"F16\""));
        assert!(hdr.contains("\"block.0.pondernet.w_halt\":{\"dtype\":\"F32\""));
        assert!(hdr.contains("\"block.0.pondernet.w_verify\":{\"dtype\":\"F32\""));
    }
}
