//! Core systems primitives for Quintessa (`GEMINI.md` compliant).
//!
//! Zero heap allocation, zero string formatting, O(1) ring buffers,
//! compact integer IDs, `#[repr(u8)]` enums, and packed bitflags.

use core::mem::MaybeUninit;

// ---------------------------------------------------------------------------
// Compact Typed IDs (Section 24)
// ---------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FeatureId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u8);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SaciOpId(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NcmNodeId(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScheduleId(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PlanId(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QueryId(pub u64);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeuronGroupId(pub u16);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DomainConceptId(pub u32);

// ---------------------------------------------------------------------------
// Enum-Only Runtime Semantics (Sections 7, 8, 9, 10)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeatureFamily {
    Knowledge = 0,
    Reasoning = 1,
    SaciInterface = 2,
    Simulation = 3,
}

impl FeatureFamily {
    #[inline(always)]
    pub const fn bit(self) -> u8 {
        1u8 << (self as u8)
    }

    pub const ALL: [FeatureFamily; 4] = [
        FeatureFamily::Knowledge,
        FeatureFamily::Reasoning,
        FeatureFamily::SaciInterface,
        FeatureFamily::Simulation,
    ];
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Module {
    Frontend = 0,
    ArchIr = 1,
    FeatureEtf = 2,
    WspStatic = 3,
    WspDynamic = 4,
    SparseLowering = 5,
    Attention = 6,
    Saci = 7,
    Workspace = 8,
    Ncm = 9,
    Runtime = 10,
    ShardedCompiler = 11,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Idle = 0,
    Ready = 1,
    Running = 2,
    Waiting = 3,
    Completed = 4,
    Failed = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    None = 0,
    QueueFull = 1,
    CapacityExceeded = 2,
    InvalidArchitecture = 3,
    InvalidAttentionTopology = 4,
    InvalidFeatureDim = 5,
    ExclusivityViolation = 6,
    IllegalPathway = 7,
    UnknownSaciOp = 8,
    SaciToleranceExceeded = 9,
    WorkspaceMaxIters = 10,
    NcmCycleDetected = 11,
    InvalidNcmGraph = 12,
    BufferTooSmall = 13,
    UnauthorizedWrite = 14,
    ShapeMismatch = 15,
    ProtectedInvariant = 16,
    SingularMatrix = 17,
    MaxPonderStepsExceeded = 18,
    VerificationFailed = 19,
    InvalidInput = 20,
    NumericOverflow = 21,
    IoError = 22,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdaComponent {
    BrainKda = 0,
    BrainMlaMoe = 1,
    BrainEmbedProj = 2,
    BrainDecoder = 3,
    FrozenVq = 4,
    SaciCircuits = 5,
    SaciRouter = 6,
    WorkspaceProjNorm = 7,
    BridgesHaltingVerify = 8,
    ConceptGenerator = 9,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WritePolicy {
    Immutable = 0,
    BypassedDoNotTarget = 1,
    EligibleWithInvariants = 2,
    EligibleTargeted = 3,
}

impl WritePolicy {
    #[inline]
    pub const fn is_writable(self) -> bool {
        matches!(
            self,
            WritePolicy::EligibleWithInvariants | WritePolicy::EligibleTargeted
        )
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SolverKind {
    ProgramToWeight = 0,
    RidgePseudoinverse = 1,
    SvdNullspace = 2,
    LowRankEdit = 3,
    TaskVector = 4,
    CmaEs = 5,
    NelderMead = 6,
    MeZo = 7,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PonderAction {
    Continue = 0,
    Halt = 1,
    EscalateToSaci = 2,
    RevisePlan = 3,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    ArchCompileStarted = 0,
    EtfAllocated = 1,
    StaticWspBuilt = 2,
    AttentionLowered = 3,
    ArchCompileCompleted = 4,
    QueryPlanValidated = 5,
    WspMasksGenerated = 6,
    SparseScheduleLowered = 7,
    SparseDispatchExecuted = 8,
    SaciKernelDispatched = 9,
    WorkspaceStep = 10,
    WorkspaceHalted = 11,
    NcmWaveScheduled = 12,
    ExclusivityChecked = 13,
    Error = 14,
    BootstrapRouted = 15,
    PlanRevised = 16,
    SaciDagValidated = 17,
    VerificationFailed = 18,
    ShardPlanBuilt = 19,
    ShardStarted = 20,
    ShardCompleted = 21,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VerificationFeedback {
    pub passed: bool,
    pub violated_constraint: u16,
    pub observed_value: f64,
    pub allowed_value: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeuronAuditRecord {
    pub query_id: QueryId,
    pub revision: u32,
    pub layer: u8,
    pub selected_mask: u8,
    pub neuron_id: u32,
    pub scores: [f32; 4],
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Kda = 0,
    Mla = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalMode {
    Exact = 0,
    CertifiedApproximate = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu = 0,
    Gpu = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceState {
    Init = 0,
    ExecutePlan = 1,
    Simulate = 2,
    Verify = 3,
    RevisePlan = 4,
    HaltedVerified = 5,
    HaltedMaxIters = 6,
}

// ---------------------------------------------------------------------------
// Bitflags (Sections 7, 20)
// ---------------------------------------------------------------------------

pub const MASK_INACTIVE: u8 = 0;
pub const MASK_KNOWLEDGE: u8 = 1 << 0;
pub const MASK_REASONING: u8 = 1 << 1;
pub const MASK_SACI: u8 = 1 << 2;
pub const MASK_SIMULATION: u8 = 1 << 3;
pub const MASK_LEGAL_ALL: u8 = MASK_KNOWLEDGE | MASK_REASONING | MASK_SACI | MASK_SIMULATION;

pub const FLAG_VALIDATED: u16 = 1 << 0;
pub const FLAG_SPARSE_SKIP: u16 = 1 << 1;
pub const FLAG_CAUSAL: u16 = 1 << 2;
pub const FLAG_VERIFIED: u16 = 1 << 3;
pub const FLAG_FAILED: u16 = 1 << 4;

// ---------------------------------------------------------------------------
// Structured Telemetry (Sections 10, 34)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelemetryEvent {
    pub tick: u64,
    pub module: Module,
    pub event: EventKind,
    pub status: Status,
    pub code: ErrorCode,
    pub flags: u16,
    pub id: u32,
    pub value: u32,
}

// ---------------------------------------------------------------------------
// Fixed-Capacity Ring Buffer (Section 6)
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueFullPolicy {
    Reject = 0,
    OverwriteOldest = 1,
}

pub struct RingBuffer<T: Copy, const CAP: usize> {
    buf: [MaybeUninit<T>; CAP],
    head: usize,
    tail: usize,
    len: usize,
    policy: QueueFullPolicy,
}

impl<T: Copy, const CAP: usize> RingBuffer<T, CAP> {
    const _ASSERT_POW2: () = assert!(CAP > 0 && (CAP & (CAP - 1)) == 0, "CAP must be power of 2");
    const MASK: usize = CAP - 1;

    pub const fn new(policy: QueueFullPolicy) -> Self {
        let () = Self::_ASSERT_POW2;
        Self {
            buf: [const { MaybeUninit::uninit() }; CAP],
            head: 0,
            tail: 0,
            len: 0,
            policy,
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len == CAP
    }

    #[inline]
    pub fn push(&mut self, item: T) -> ErrorCode {
        if self.len == CAP {
            match self.policy {
                QueueFullPolicy::Reject => return ErrorCode::QueueFull,
                QueueFullPolicy::OverwriteOldest => {
                    self.head = (self.head + 1) & Self::MASK;
                    self.len -= 1;
                }
            }
        }
        self.buf[self.tail].write(item);
        self.tail = (self.tail + 1) & Self::MASK;
        self.len += 1;
        ErrorCode::None
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        // SAFETY: `head` points to an initialized slot because `len > 0`.
        let val = unsafe { self.buf[self.head].assume_init_read() };
        self.head = (self.head + 1) & Self::MASK;
        self.len -= 1;
        Some(val)
    }
}

pub struct TelemetryBus<const CAP: usize> {
    ring: RingBuffer<TelemetryEvent, CAP>,
    tick: u64,
    pub dropped_count: u64,
}

impl<const CAP: usize> TelemetryBus<CAP> {
    pub const fn new() -> Self {
        Self {
            ring: RingBuffer::new(QueueFullPolicy::OverwriteOldest),
            tick: 0,
            dropped_count: 0,
        }
    }

    #[inline]
    pub fn emit(
        &mut self,
        module: Module,
        event: EventKind,
        status: Status,
        code: ErrorCode,
        flags: u16,
        id: u32,
        value: u32,
    ) {
        self.tick = self.tick.wrapping_add(1);
        if self.ring.is_full() {
            self.dropped_count += 1;
        }
        self.ring.push(TelemetryEvent {
            tick: self.tick,
            module,
            event,
            status,
            code,
            flags,
            id,
            value,
        });
    }

    #[inline]
    pub fn pop(&mut self) -> Option<TelemetryEvent> {
        self.ring.pop()
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.ring.len()
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

impl<const CAP: usize> Default for TelemetryBus<CAP> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fixed-Capacity ArrayVec (Sections 3, 5, 27)
// ---------------------------------------------------------------------------

pub struct ArrayVec<T: Copy, const CAP: usize> {
    buf: [MaybeUninit<T>; CAP],
    len: usize,
}

impl<T: Copy, const CAP: usize> ArrayVec<T, CAP> {
    pub const fn new() -> Self {
        Self {
            buf: [const { MaybeUninit::uninit() }; CAP],
            len: 0,
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn push(&mut self, item: T) -> ErrorCode {
        if self.len >= CAP {
            return ErrorCode::CapacityExceeded;
        }
        self.buf[self.len].write(item);
        self.len += 1;
        ErrorCode::None
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: slot `self.len` was initialized.
        Some(unsafe { self.buf[self.len].assume_init_read() })
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: first `self.len` elements are initialized.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr() as *const T, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: first `self.len` elements are initialized.
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut T, self.len) }
    }
}

impl<T: Copy, const CAP: usize> Default for ArrayVec<T, CAP> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const CAP: usize> Clone for ArrayVec<T, CAP> {
    fn clone(&self) -> Self {
        let mut out = Self::new();
        for &item in self.as_slice() {
            let _ = out.push(item);
        }
        out
    }
}

impl<T: Copy + core::fmt::Debug, const CAP: usize> core::fmt::Debug for ArrayVec<T, CAP> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}
