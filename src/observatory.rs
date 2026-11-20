//! Observatory: Presentation Layer Boundary (Sections 11, 12, 35).
//!
//! All human-readable string conversions and formatting live exclusively here.
//! Core compiler, IR, WSP, and runtime modules never construct or format strings.

use crate::sys::{
    AttentionKind, ErrorCode, EventKind, FeatureFamily, Module, Status, TelemetryEvent,
    WorkspaceState,
};

pub fn module_name(m: Module) -> &'static str {
    match m {
        Module::Frontend => "Frontend",
        Module::ArchIr => "ArchIR",
        Module::FeatureEtf => "FeatureETF",
        Module::WspStatic => "WspStatic",
        Module::WspDynamic => "WspDynamic",
        Module::SparseLowering => "SparseLowering",
        Module::Attention => "Attention",
        Module::Saci => "SACI",
        Module::Workspace => "Workspace",
        Module::Ncm => "NCM",
        Module::Runtime => "Runtime",
        Module::ShardedCompiler => "ShardedCompiler",
    }
}

pub fn family_name(f: FeatureFamily) -> &'static str {
    match f {
        FeatureFamily::Knowledge => "Knowledge",
        FeatureFamily::Reasoning => "Reasoning",
        FeatureFamily::SaciInterface => "SACI-Interface",
        FeatureFamily::Simulation => "Simulation",
    }
}

pub fn attention_name(k: AttentionKind) -> &'static str {
    match k {
        AttentionKind::Kda => "KDA",
        AttentionKind::Mla => "MLA",
    }
}

pub fn workspace_state_name(s: WorkspaceState) -> &'static str {
    match s {
        WorkspaceState::Init => "Init",
        WorkspaceState::ExecutePlan => "ExecutePlan",
        WorkspaceState::Simulate => "Simulate",
        WorkspaceState::Verify => "Verify",
        WorkspaceState::RevisePlan => "RevisePlan",
        WorkspaceState::HaltedVerified => "Halted(Verified)",
        WorkspaceState::HaltedMaxIters => "Halted(MaxIters)",
    }
}

pub fn status_name(s: Status) -> &'static str {
    match s {
        Status::Idle => "Idle",
        Status::Ready => "Ready",
        Status::Running => "Running",
        Status::Waiting => "Waiting",
        Status::Completed => "Completed",
        Status::Failed => "Failed",
    }
}

pub fn error_description(e: ErrorCode) -> &'static str {
    match e {
        ErrorCode::None => "No error",
        ErrorCode::QueueFull => "Ring buffer exhausted",
        ErrorCode::CapacityExceeded => "Fixed buffer capacity exceeded",
        ErrorCode::InvalidArchitecture => "Invalid ADA architecture specification",
        ErrorCode::InvalidAttentionTopology => "Attention topology must be 6 KDA + 2 MLA",
        ErrorCode::InvalidFeatureDim => "Invalid ETF feature or substrate dimension",
        ErrorCode::ExclusivityViolation => "Neuron activated >1 feature family (sum z_{i,f} > 1)",
        ErrorCode::IllegalPathway => "Query requested illegal neuron/feature pathway",
        ErrorCode::UnknownSaciOp => "Unknown SACI operation ID",
        ErrorCode::SaciToleranceExceeded => "SACI kernel exceeded declared error tolerance",
        ErrorCode::WorkspaceMaxIters => "Workspace reached max iterations without verification",
        ErrorCode::NcmCycleDetected => "Cycle detected in NCM output dependency graph",
        ErrorCode::InvalidNcmGraph => "Invalid NCM output graph CSR layout",
        ErrorCode::BufferTooSmall => "Preallocated scratch buffer too small",
        ErrorCode::UnauthorizedWrite => "Unauthorized weight-space tensor write",
        ErrorCode::ShapeMismatch => "Tensor or matrix shape mismatch",
        ErrorCode::ProtectedInvariant => "Protected invariant tensor cannot be modified",
        ErrorCode::SingularMatrix => "Singular matrix in analytical WSP solver",
        ErrorCode::MaxPonderStepsExceeded => "Adaptive PonderNet exceeded max steps",
        ErrorCode::VerificationFailed => "Candidate checkpoint verification failed",
        ErrorCode::InvalidInput => "Invalid input to compiler or WSP solver",
        ErrorCode::NumericOverflow => "Checked 64-bit parameter or byte offset overflow",
        ErrorCode::IoError => "Filesystem I/O error during streaming shard emission",
    }
}

pub fn event_description(ev: EventKind) -> &'static str {
    match ev {
        EventKind::ArchCompileStarted => "Pipeline A: architecture compilation started",
        EventKind::EtfAllocated => "Feature/ETF 4-family basis allocated",
        EventKind::StaticWspBuilt => "Static WSP pathway metadata built",
        EventKind::AttentionLowered => "6 KDA + 2 MLA attention layers lowered",
        EventKind::ArchCompileCompleted => "Pipeline A: architecture compilation completed",
        EventKind::QueryPlanValidated => "Pipeline B: query plan validated",
        EventKind::WspMasksGenerated => "Dynamic WSP per-neuron masks generated",
        EventKind::SparseScheduleLowered => "Sparse execution schedule lowered",
        EventKind::SparseDispatchExecuted => "Sparse feature dispatch executed",
        EventKind::SaciKernelDispatched => "SACI deterministic kernel dispatched",
        EventKind::WorkspaceStep => "Workspace state transition",
        EventKind::WorkspaceHalted => "Workspace loop halted",
        EventKind::NcmWaveScheduled => "NCM parallel output wave scheduled",
        EventKind::ExclusivityChecked => "Neuron mutual-exclusion invariant checked",
        EventKind::Error => "Error occurred",
        EventKind::BootstrapRouted => "Phase 1: Bootstrap Knowledge+Reasoning routing completed",
        EventKind::PlanRevised => "Phase 6: Plan revised after verification feedback",
        EventKind::SaciDagValidated => "Phase 4: SACI execution DAG validated",
        EventKind::VerificationFailed => "Workspace verification failed; triggering replanning",
        EventKind::ShardPlanBuilt => "Trillion-scale sharded compilation plan built",
        EventKind::ShardStarted => "Streaming shard synthesis started",
        EventKind::ShardCompleted => "Streaming shard synthesis completed and committed",
    }
}

pub fn format_event(ev: &TelemetryEvent) -> String {
    format!(
        "[tick={}] [{}] {} ({}) code={} flags=0x{:04x} id={} val={}",
        ev.tick,
        module_name(ev.module),
        event_description(ev.event),
        status_name(ev.status),
        error_description(ev.code),
        ev.flags,
        ev.id,
        ev.value,
    )
}

pub fn format_verifier_report(r: &crate::verifier::VerifierReport) -> String {
    let status_str = match r.status {
        crate::verifier::VerifyStatus::Pass => "PASS",
        crate::verifier::VerifyStatus::Fail => "FAIL",
    };
    format!(
        "Quintessa Verifier [{}] (d={}, m={}, rules={}, outputs={}, k={}, trials={})\n\
         - Activation: mean_active={:.2}, sparsity={:.2}%, forced_trigger={:.2}%, leakage={:.4}%, dead_rules={}\n\
         - Routing:    exclusivity_violations={}, active_ratio={:.4}, exact_route_acc={:.2}%\n\
         - Accuracy:   rand_err={:.3}%, forced_err={:.3}%, bnd_miss_pos_err={:.3}%, bnd_pres_neg_err={:.3}%, agreement={:.2}%\n\
         - ETF:        max_coh={:.4}, rms_coh={:.4}, welch={:.4}, recall={:.2}%, absent_fpr={:.3}%, safety_z={:.2}\n\
         - Noise:      measured_std={:.4}, theoretical_std={:.4}, snr={:.2} dB, max_abs={:.4}, rec_min_d={}",
        status_str,
        r.hidden_dim,
        r.num_features,
        r.num_rules,
        r.num_outputs,
        r.active_k,
        r.trials,
        r.activation.mean_active_rules,
        r.activation.activation_sparsity * 100.0,
        r.activation.forced_fire_trigger_rate * 100.0,
        r.activation.non_firing_leakage_rate * 100.0,
        r.activation.dead_rule_count,
        r.routing.exclusivity_violations,
        r.routing.active_neuron_ratio,
        r.routing.routing_accuracy * 100.0,
        r.accuracy.random_sparse_error * 100.0,
        r.accuracy.forced_fire_error * 100.0,
        r.accuracy.boundary_missing_pos_error * 100.0,
        r.accuracy.boundary_present_neg_error * 100.0,
        r.accuracy.overall_agreement * 100.0,
        r.etf.max_pairwise_coherence,
        r.etf.rms_pairwise_coherence,
        r.etf.welch_bound,
        r.etf.present_feature_recall * 100.0,
        r.etf.absent_feature_fpr * 100.0,
        r.etf.safety_factor_z,
        r.noise.measured_interference_std,
        r.noise.theoretical_interference_std,
        r.noise.snr_db,
        r.noise.max_abs_interference,
        r.noise.recommended_min_width,
    )
}
