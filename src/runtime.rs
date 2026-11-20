//! Pipeline A (Architecture Compilation) and Pipeline B (Per-Query Execution).
//!
//! Pipeline B operates with zero runtime heap allocation using preallocated
//! `AdaQueryScratch`.

use crate::attention::{run_attention_stack, AttentionScratch};
use crate::capability::{CapabilityGraph, CapabilitySpec};
use crate::etf::compile_feature_etf;
use crate::ir::{
    ArchitectureIr, ArchitectureSpec, AttentionIr, NcmOutputIr, SaciDag, SaciIr, WorkflowPlanIr,
    ADA_ATTENTION_TOPOLOGY,
};
use crate::manifest::{lower_capabilities, FeatureNeuronManifest};
use crate::sys::{
    AttentionKind, ErrorCode, EventKind, Module, NcmNodeId, SaciOpId, Status, TelemetryBus,
    VerificationFeedback, WorkspaceState, FLAG_FAILED, FLAG_VALIDATED, FLAG_VERIFIED,
};
use crate::wsp::{
    bootstrap_wsp, execute_sparse_substrate, generate_wsp_masks, lower_sparse_schedule,
    SparseStats, StaticWspMetadata, WspScratch,
};

#[derive(Clone, Debug)]
pub struct CompiledAdaModel {
    pub ir: ArchitectureIr,
    pub static_wsp: StaticWspMetadata,
}

/// Pipeline A: Compile an `ArchitectureSpec` + `SaciIr` using the default 6 KDA + 2 MLA topology.
pub fn compile_architecture<const CAP: usize>(
    spec: &ArchitectureSpec,
    saci: SaciIr,
    legal_masks: Option<&[u8]>,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<CompiledAdaModel, ErrorCode> {
    compile_architecture_with_topology(spec, &ADA_ATTENTION_TOPOLOGY, saci, legal_masks, telemetry)
}

/// Pipeline A: Compile an `ArchitectureSpec` + dynamic `topology` + `SaciIr` into `CompiledAdaModel`.
pub fn compile_architecture_with_topology<const CAP: usize>(
    spec: &ArchitectureSpec,
    topology: &[AttentionKind],
    saci: SaciIr,
    legal_masks: Option<&[u8]>,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<CompiledAdaModel, ErrorCode> {
    telemetry.emit(
        Module::ArchIr,
        EventKind::ArchCompileStarted,
        Status::Running,
        ErrorCode::None,
        0,
        spec.num_neurons as u32,
        spec.substrate_dim as u32,
    );

    if spec.vq_codebook_size == 0 || spec.tile_size == 0 {
        return Err(ErrorCode::InvalidArchitecture);
    }

    let attention = AttentionIr::build(topology, spec.kda, spec.mla)?;
    telemetry.emit(
        Module::Attention,
        EventKind::AttentionLowered,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        topology.len() as u32,
        0,
    );

    let feature_etf = compile_feature_etf(spec.num_neurons, spec.substrate_dim, legal_masks)?;
    telemetry.emit(
        Module::FeatureEtf,
        EventKind::EtfAllocated,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        spec.num_neurons as u32,
        spec.substrate_dim as u32,
    );

    let ir = ArchitectureIr {
        vq_codebook_size: spec.vq_codebook_size,
        tile_size: spec.tile_size,
        feature_etf,
        attention,
        saci,
        workspace: spec.workspace,
    };

    let static_wsp = StaticWspMetadata::from_arch_ir(&ir);
    telemetry.emit(
        Module::WspStatic,
        EventKind::StaticWspBuilt,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        static_wsp.num_tiles as u32,
        static_wsp.legal_saci_ops.len() as u32,
    );

    telemetry.emit(
        Module::ArchIr,
        EventKind::ArchCompileCompleted,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        spec.num_neurons as u32,
        spec.substrate_dim as u32,
    );

    Ok(CompiledAdaModel { ir, static_wsp })
}

/// Domain Architecture Orchestrator:
/// `CapabilitySpec[] -> CapabilityGraph -> FeatureNeuronManifest -> Pipeline A`.
///
/// Validates SACI bindings and materializes domain-specialized `legal_masks`
/// into `CompiledAdaModel` without rewriting Pipeline A.
pub fn compile_domain_architecture<const CAP: usize>(
    spec: &ArchitectureSpec,
    topology: &[AttentionKind],
    capabilities: &[CapabilitySpec],
    saci: SaciIr,
    neurons_per_capability: usize,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(CompiledAdaModel, CapabilityGraph, FeatureNeuronManifest), ErrorCode> {
    let graph = CapabilityGraph::compile(capabilities)?;
    let manifest = lower_capabilities(
        &graph,
        capabilities,
        spec.num_neurons,
        neurons_per_capability,
    )?;

    for binding in &manifest.saci_bindings {
        if saci.lookup(binding.op_id).is_none() {
            return Err(ErrorCode::UnknownSaciOp);
        }
    }

    let model = compile_architecture_with_topology(
        spec,
        topology,
        saci,
        Some(&manifest.legal_masks),
        telemetry,
    )?;

    Ok((model, graph, manifest))
}

// ---------------------------------------------------------------------------
// Preallocated Scratch for Pipeline B (Zero Runtime Allocation)
// ---------------------------------------------------------------------------

pub struct AdaQueryScratch {
    pub wsp: WspScratch,
    pub attn: AttentionScratch,
    pub attn_out: Vec<f64>,
    pub substrate_out: Vec<f64>,
    pub saci_in_deg: Vec<u16>,
    pub saci_wave_buf: Vec<u16>,
    pub saci_next_wave: Vec<u16>,
    pub ncm_in_deg: Vec<u16>,
    pub ncm_wave_buf: Vec<NcmNodeId>,
    pub ncm_next_wave: Vec<NcmNodeId>,
    pub ncm_wave_counts: Vec<u16>,
    pub ncm_order: Vec<NcmNodeId>,
}

impl AdaQueryScratch {
    pub fn new(model: &CompiledAdaModel, max_ncm_nodes: usize) -> Self {
        Self::with_max_seq_len(model, max_ncm_nodes, 64)
    }

    pub fn with_max_seq_len(
        model: &CompiledAdaModel,
        max_ncm_nodes: usize,
        max_seq_len: usize,
    ) -> Self {
        let dim = model.ir.feature_etf.substrate_dim;
        let max_saci = model.ir.saci.ops.len().max(16);
        Self {
            wsp: WspScratch::for_static(&model.static_wsp),
            attn: AttentionScratch::new(&model.ir.attention, dim, max_seq_len),
            attn_out: vec![0.0; dim],
            substrate_out: vec![0.0; dim],
            saci_in_deg: vec![0; max_saci],
            saci_wave_buf: vec![0; max_saci],
            saci_next_wave: vec![0; max_saci],
            ncm_in_deg: vec![0; max_ncm_nodes],
            ncm_wave_buf: vec![NcmNodeId(0); max_ncm_nodes],
            ncm_next_wave: vec![NcmNodeId(0); max_ncm_nodes],
            ncm_wave_counts: vec![0; max_ncm_nodes],
            ncm_order: vec![NcmNodeId(0); max_ncm_nodes],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct QueryExecutionResult {
    pub sparse_stats: SparseStats,
    pub workspace_state: WorkspaceState,
    pub workspace_iters: u16,
    pub ncm_num_waves: usize,
    pub ncm_total_nodes: usize,
    pub saci_dispatched: usize,
}

/// Deterministic SACI kernel dispatch (verifies contract & tolerance).
pub fn dispatch_saci_op<const CAP: usize>(
    saci: &SaciIr,
    op_id: SaciOpId,
    measured_residual: f64,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(), ErrorCode> {
    let spec = saci.lookup(op_id).ok_or(ErrorCode::UnknownSaciOp)?;
    if measured_residual > spec.tolerance {
        telemetry.emit(
            Module::Saci,
            EventKind::Error,
            Status::Failed,
            ErrorCode::SaciToleranceExceeded,
            0,
            op_id.0 as u32,
            0,
        );
        return Err(ErrorCode::SaciToleranceExceeded);
    }
    telemetry.emit(
        Module::Saci,
        EventKind::SaciKernelDispatched,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        op_id.0 as u32,
        spec.backend as u32,
    );
    Ok(())
}

/// Schedule NCM output graph into parallel topological waves using preallocated scratch.
pub fn schedule_ncm_waves<const CAP: usize>(
    ncm: &NcmOutputIr,
    scratch: &mut AdaQueryScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<(usize, usize), ErrorCode> {
    let n = ncm.num_nodes;
    if scratch.ncm_in_deg.len() < n {
        return Err(ErrorCode::BufferTooSmall);
    }

    scratch.ncm_in_deg[..n].copy_from_slice(&ncm.in_degrees);
    let mut wave_len = 0usize;
    for i in 0..n {
        if scratch.ncm_in_deg[i] == 0 {
            scratch.ncm_wave_buf[wave_len] = NcmNodeId(i as u16);
            wave_len += 1;
        }
    }

    let mut num_waves = 0usize;
    let mut order_len = 0usize;

    while wave_len > 0 {
        scratch.ncm_wave_counts[num_waves] = wave_len as u16;
        let mut next_len = 0usize;

        for w in 0..wave_len {
            let u = scratch.ncm_wave_buf[w];
            scratch.ncm_order[order_len] = u;
            order_len += 1;

            for &v in ncm.dependents(u) {
                let d = &mut scratch.ncm_in_deg[v.0 as usize];
                *d -= 1;
                if *d == 0 {
                    scratch.ncm_next_wave[next_len] = v;
                    next_len += 1;
                }
            }
        }

        telemetry.emit(
            Module::Ncm,
            EventKind::NcmWaveScheduled,
            Status::Completed,
            ErrorCode::None,
            0,
            num_waves as u32,
            wave_len as u32,
        );

        num_waves += 1;
        scratch.ncm_wave_buf[..next_len].copy_from_slice(&scratch.ncm_next_wave[..next_len]);
        wave_len = next_len;
    }

    if order_len != n {
        return Err(ErrorCode::NcmCycleDetected);
    }

    Ok((num_waves, order_len))
}

/// Phase 4: Validate and execute a `SaciDag` in topological waves using `AdaQueryScratch`.
pub fn execute_saci_dag<const CAP: usize>(
    saci: &SaciIr,
    dag: &SaciDag,
    scratch: &mut AdaQueryScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<usize, ErrorCode> {
    let n = dag.ops.len();
    if n == 0 {
        return Ok(0);
    }
    if scratch.saci_in_deg.len() < n {
        return Err(ErrorCode::BufferTooSmall);
    }
    for &op_id in &dag.ops {
        if saci.lookup(op_id).is_none() {
            return Err(ErrorCode::UnknownSaciOp);
        }
    }

    telemetry.emit(
        Module::Saci,
        EventKind::SaciDagValidated,
        Status::Completed,
        ErrorCode::None,
        FLAG_VALIDATED,
        n as u32,
        0,
    );

    scratch.saci_in_deg[..n].copy_from_slice(&dag.in_degrees);
    let mut wave_len = 0usize;
    for i in 0..n {
        if scratch.saci_in_deg[i] == 0 {
            scratch.saci_wave_buf[wave_len] = i as u16;
            wave_len += 1;
        }
    }

    let mut executed = 0usize;
    while wave_len > 0 {
        let mut next_len = 0usize;
        for w in 0..wave_len {
            let u = scratch.saci_wave_buf[w];
            dispatch_saci_op(saci, dag.ops[u as usize], 0.0, telemetry)?;
            executed += 1;
            for &v in dag.dependents(u) {
                let d = &mut scratch.saci_in_deg[v as usize];
                *d -= 1;
                if *d == 0 {
                    scratch.saci_next_wave[next_len] = v;
                    next_len += 1;
                }
            }
        }
        scratch.saci_wave_buf[..next_len].copy_from_slice(&scratch.saci_next_wave[..next_len]);
        wave_len = next_len;
    }

    Ok(executed)
}

/// Full 7-Phase Query Routing with Bootstrap + 6 KDA / 2 MLA Attention +
/// Feedback-Driven Replanning (`M_q^{(r)} -> M_q^{(r+1)}`) + SACI DAG + NCM.
pub fn execute_query_with_replanning<F, const CAP: usize>(
    model: &CompiledAdaModel,
    vq_code: u32,
    bootstrap_budget: usize,
    initial_plan: &WorkflowPlanIr<'_>,
    saci_dag: &SaciDag,
    ncm: &NcmOutputIr,
    input_vec: &[f64],
    mut revise_plan: F,
    scratch: &mut AdaQueryScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<QueryExecutionResult, ErrorCode>
where
    F: FnMut(u32, VerificationFeedback) -> Option<WorkflowPlanIr<'static>>,
{
    // Phase 1: Bootstrap (Knowledge + Reasoning only)
    let _boot_active = bootstrap_wsp(
        &model.static_wsp,
        vq_code,
        bootstrap_budget,
        &mut scratch.wsp,
        telemetry,
    )?;

    // Reset attention recurrent state for this query
    scratch.attn.reset();

    let mut current_plan = *initial_plan;
    let mut revision = 0u32;
    let mut saci_dispatched = 0usize;
    let mut last_stats = SparseStats {
        total_neurons: model.static_wsp.num_neurons as u32,
        active_neurons: 0,
        skipped_neurons: model.static_wsp.num_neurons as u32,
        total_tiles: model.static_wsp.num_tiles as u32,
        active_tiles: 0,
        total_flops: 0,
        active_flops: 0,
        skipped_flops: 0,
    };

    let mut state = WorkspaceState::Init;
    let mut iters = 0u16;

    while iters < model.ir.workspace.max_iters {
        iters += 1;
        // Phase 3 & 5: WSP mask generation + Sparse lowering
        generate_wsp_masks(
            &model.static_wsp,
            &current_plan,
            &mut scratch.wsp,
            telemetry,
        )?;
        let _ = lower_sparse_schedule(&model.static_wsp, &mut scratch.wsp, telemetry);

        state = WorkspaceState::ExecutePlan;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceStep,
            Status::Running,
            ErrorCode::None,
            0,
            iters as u32,
            state as u32,
        );

        // Execute 6 KDA + 2 MLA attention stack
        run_attention_stack(
            &model.ir.attention,
            (iters - 1) as usize,
            input_vec,
            &mut scratch.attn_out,
            &mut scratch.attn,
        )?;

        // Execute sparse superposed substrate (skips inactive neurons)
        last_stats = execute_sparse_substrate(
            &model.ir.feature_etf,
            &scratch.wsp,
            input_vec,
            &mut scratch.substrate_out,
            telemetry,
        )?;

        // Phase 4: Execute SACI DAG
        saci_dispatched += execute_saci_dag(&model.ir.saci, saci_dag, scratch, telemetry)?;

        // Phase 6: Simulate + Verify
        state = WorkspaceState::Simulate;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceStep,
            Status::Running,
            ErrorCode::None,
            0,
            iters as u32,
            state as u32,
        );
        let mut norm_sq = 0.0;
        for &v in &scratch.substrate_out {
            norm_sq += v * v;
        }

        state = WorkspaceState::Verify;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceStep,
            Status::Running,
            ErrorCode::None,
            0,
            iters as u32,
            state as u32,
        );
        if norm_sq >= current_plan.verify_threshold {
            state = WorkspaceState::HaltedVerified;
            telemetry.emit(
                Module::Workspace,
                EventKind::WorkspaceHalted,
                Status::Completed,
                ErrorCode::None,
                FLAG_VERIFIED,
                iters as u32,
                state as u32,
            );
            break;
        }

        // Verification failed -> emit feedback and trigger Reasoning replanning
        let feedback = VerificationFeedback {
            passed: false,
            violated_constraint: 1,
            observed_value: norm_sq,
            allowed_value: current_plan.verify_threshold,
        };
        telemetry.emit(
            Module::Workspace,
            EventKind::VerificationFailed,
            Status::Running,
            ErrorCode::None,
            FLAG_FAILED,
            iters as u32,
            revision,
        );

        state = WorkspaceState::RevisePlan;
        if let Some(next_plan) = revise_plan(revision, feedback) {
            revision += 1;
            current_plan = next_plan;
            telemetry.emit(
                Module::WspDynamic,
                EventKind::PlanRevised,
                Status::Running,
                ErrorCode::None,
                0,
                revision,
                iters as u32,
            );
        } else {
            break;
        }
    }

    if state != WorkspaceState::HaltedVerified {
        state = WorkspaceState::HaltedMaxIters;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceHalted,
            Status::Completed,
            ErrorCode::WorkspaceMaxIters,
            0,
            iters as u32,
            state as u32,
        );
    }

    // Phase 7: NCM parallel wave scheduling
    let (ncm_num_waves, ncm_total_nodes) = schedule_ncm_waves(ncm, scratch, telemetry)?;

    Ok(QueryExecutionResult {
        sparse_stats: last_stats,
        workspace_state: state,
        workspace_iters: iters,
        ncm_num_waves,
        ncm_total_nodes,
        saci_dispatched,
    })
}

/// Pipeline B: Execute a per-query `WorkflowPlanIr` with zero runtime heap allocation.
pub fn execute_query_pipeline_b<const CAP: usize>(
    model: &CompiledAdaModel,
    plan: &WorkflowPlanIr<'_>,
    ncm: &NcmOutputIr,
    input_vec: &[f64],
    scratch: &mut AdaQueryScratch,
    telemetry: &mut TelemetryBus<CAP>,
) -> Result<QueryExecutionResult, ErrorCode> {
    // 1. Dynamic WSP mask generation + mutual exclusion enforcement
    generate_wsp_masks(&model.static_wsp, plan, &mut scratch.wsp, telemetry)?;

    // 2. Lower masks to genuinely sparse active-neuron + tile schedule
    let sparse_stats = lower_sparse_schedule(&model.static_wsp, &mut scratch.wsp, telemetry);

    // 3. Workspace iterative execution loop (ExecutePlan -> Simulate -> Verify)
    let mut state = WorkspaceState::Init;
    let mut iters = 0u16;
    let mut saci_dispatched = 0usize;
    let mut last_stats = sparse_stats;

    while iters < model.ir.workspace.max_iters {
        iters += 1;
        state = WorkspaceState::ExecutePlan;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceStep,
            Status::Running,
            ErrorCode::None,
            0,
            iters as u32,
            state as u32,
        );

        // Sparse neural substrate execution (skips inactive neurons)
        last_stats = execute_sparse_substrate(
            &model.ir.feature_etf,
            &scratch.wsp,
            input_vec,
            &mut scratch.substrate_out,
            telemetry,
        )?;

        // Dispatch requested SACI kernels
        for &op_id in plan.saci_ops {
            dispatch_saci_op(&model.ir.saci, op_id, 0.0, telemetry)?;
            saci_dispatched += 1;
        }

        state = WorkspaceState::Simulate;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceStep,
            Status::Running,
            ErrorCode::None,
            0,
            iters as u32,
            state as u32,
        );
        let mut norm_sq = 0.0;
        for &v in &scratch.substrate_out {
            norm_sq += v * v;
        }

        state = WorkspaceState::Verify;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceStep,
            Status::Running,
            ErrorCode::None,
            0,
            iters as u32,
            state as u32,
        );
        if norm_sq >= plan.verify_threshold {
            state = WorkspaceState::HaltedVerified;
            telemetry.emit(
                Module::Workspace,
                EventKind::WorkspaceHalted,
                Status::Completed,
                ErrorCode::None,
                FLAG_VERIFIED,
                iters as u32,
                state as u32,
            );
            break;
        }
        state = WorkspaceState::RevisePlan;
    }

    if state != WorkspaceState::HaltedVerified {
        state = WorkspaceState::HaltedMaxIters;
        telemetry.emit(
            Module::Workspace,
            EventKind::WorkspaceHalted,
            Status::Completed,
            ErrorCode::WorkspaceMaxIters,
            0,
            iters as u32,
            state as u32,
        );
    }

    // 4. NCM dependency-aware parallel wave scheduling
    let (ncm_num_waves, ncm_total_nodes) = schedule_ncm_waves(ncm, scratch, telemetry)?;

    Ok(QueryExecutionResult {
        sparse_stats: last_stats,
        workspace_state: state,
        workspace_iters: iters,
        ncm_num_waves,
        ncm_total_nodes,
        saci_dispatched,
    })
}
