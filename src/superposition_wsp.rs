//! Quintessa Weight-Space Programming in Superposition (`superposition_wsp`).
//!
//! Compiles deterministic software engine logic directly into superposed neural weights:
//! 1. `FeatureSpace` — near-orthogonal unit directions `U in R^{m x d}` (Gaussian or ETF-seeded)
//! 2. `RuleLayer` — AND / NOT / OR boolean rules via shifted ReLU `a = max(W h - b, 0) / (1 - tau)`
//!    and sparse-active SAXPY write `h_out = sum_{a_r > 0} a_r * v_r`
//! 3. `NumericThresholdLayer` — ordinary threshold units `f = clip(g * (x - t) + 0.5, 0, 1)`
//!    projected into superposition directions
//! 4. `LinearScorerLayer` — exact weighted linear combinations `sum_j c_j a_j + bias`
//! 5. `DecisionTree` — compiled leaf-by-leaf into AND/NOT path rules in a `RuleLayer`
//! 6. `SuperpositionStack` — multi-layer chaining where layer L+1 reads layer L's output space
//!
//! Hot-path inference (`write_into`, `forward_into`, `dot`) is strictly zero-allocation,
//! integer-indexed, and uses preallocated `SuperpositionScratch`.

use crate::sys::ErrorCode;

/// Deterministic SplitMix64 PRNG for reproducible direction codes (stored as seed in metadata).
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.state;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)` open interval (safe for Box-Muller `ln`).
    #[inline]
    pub fn next_f64_open(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) | 1;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Standard normal `N(0, 1)` via Box-Muller transform.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }
}

/// d-dimensional hidden space holding `num_features` near-orthogonal unit directions `U[m, d]`.
/// Feature index `0` may optionally be reserved as the `__one__` always-on sentinel when needed.
#[derive(Clone, Debug)]
pub struct FeatureSpace {
    pub num_features: usize,
    pub dim: usize,
    pub seed: u64,
    /// Row-major `[num_features, dim]` unit vectors.
    pub directions: Vec<f64>,
}

impl FeatureSpace {
    /// Generate `num_features` random Gaussian unit directions in `R^dim` from `seed`.
    pub fn new_gaussian(num_features: usize, dim: usize, seed: u64) -> Result<Self, ErrorCode> {
        if num_features == 0 || dim == 0 {
            return Err(ErrorCode::InvalidFeatureDim);
        }
        let mut rng = SplitMix64::new(seed);
        let mut directions = vec![0.0_f64; num_features * dim];
        for f in 0..num_features {
            let row = &mut directions[f * dim..(f + 1) * dim];
            let mut norm_sq = 0.0_f64;
            for v in row.iter_mut() {
                let g = rng.next_normal();
                *v = g;
                norm_sq += g * g;
            }
            let inv = if norm_sq > 1e-30 {
                1.0 / norm_sq.sqrt()
            } else {
                0.0
            };
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        Ok(Self {
            num_features,
            dim,
            seed,
            directions,
        })
    }

    #[inline]
    pub fn direction(&self, feature_idx: u32) -> Option<&[f64]> {
        let idx = feature_idx as usize;
        if idx >= self.num_features {
            return None;
        }
        let start = idx * self.dim;
        Some(&self.directions[start..start + self.dim])
    }

    /// Encode active `(feature_idx, amplitude)` pairs into preallocated `h_out` (zero-allocation).
    #[inline]
    pub fn write_into(&self, active: &[(u32, f64)], h_out: &mut [f64]) -> Result<(), ErrorCode> {
        if h_out.len() < self.dim {
            return Err(ErrorCode::BufferTooSmall);
        }
        for v in h_out[..self.dim].iter_mut() {
            *v = 0.0;
        }
        for &(f_idx, amp) in active {
            let dir = self.direction(f_idx).ok_or(ErrorCode::InvalidInput)?;
            for (dst, &u) in h_out[..self.dim].iter_mut().zip(dir.iter()) {
                *dst += amp * u;
            }
        }
        Ok(())
    }

    /// Readout dot product `h · u_j` (zero-allocation).
    #[inline]
    pub fn dot(&self, h: &[f64], feature_idx: u32) -> Result<f64, ErrorCode> {
        if h.len() < self.dim {
            return Err(ErrorCode::BufferTooSmall);
        }
        let dir = self.direction(feature_idx).ok_or(ErrorCode::InvalidInput)?;
        let mut acc = 0.0_f64;
        for (&hv, &uv) in h[..self.dim].iter().zip(dir.iter()) {
            acc += hv * uv;
        }
        Ok(acc)
    }

    /// Boolean threshold readout `h · u_j > threshold`.
    #[inline]
    pub fn read_bool(
        &self,
        h: &[f64],
        feature_idx: u32,
        threshold: f64,
    ) -> Result<bool, ErrorCode> {
        Ok(self.dot(h, feature_idx)? > threshold)
    }

    /// Compute maximum and RMS pairwise coherence `|<u_i, u_j>|` (`i != j`) over up to `sample_limit`
    /// features (exact when `num_features <= sample_limit`).
    pub fn pairwise_coherence(&self, sample_limit: usize) -> (f64, f64) {
        let n = self.num_features.min(sample_limit);
        if n < 2 {
            return (0.0, 0.0);
        }
        let mut max_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut pairs = 0u64;
        for i in 0..n {
            let ui = &self.directions[i * self.dim..(i + 1) * self.dim];
            for j in (i + 1)..n {
                let uj = &self.directions[j * self.dim..(j + 1) * self.dim];
                let mut d = 0.0_f64;
                for (&a, &b) in ui.iter().zip(uj.iter()) {
                    d += a * b;
                }
                let ad = d.abs();
                if ad > max_abs {
                    max_abs = ad;
                }
                sum_sq += d * d;
                pairs += 1;
            }
        }
        let rms = if pairs > 0 {
            (sum_sq / (pairs as f64)).sqrt()
        } else {
            0.0
        };
        (max_abs, rms)
    }

    /// Theoretical Welch bound for `m` unit vectors in `R^d`: `sqrt((m - d) / (d * (m - 1)))`.
    #[inline]
    pub fn welch_bound(&self) -> f64 {
        if self.num_features <= self.dim || self.num_features <= 1 {
            return 0.0;
        }
        let m = self.num_features as f64;
        let d = self.dim as f64;
        ((m - d) / (d * (m - 1.0))).sqrt()
    }

    /// Compute oblique interference-cancelling read directions (inspired by Gemma Scope 2's
    /// asymmetric `W_enc != W_dec^T` finding):
    ///   `u_tilde_j = (U^T U + lambda * I_d)^{-1} u_j`, normalized so `u_tilde_j · u_j == 1.0`.
    /// When `num_features <= dim` and `lambda` is small, cross-feature interference vanishes.
    pub fn compute_oblique_read_directions(&self, lambda: f64) -> Result<FeatureSpace, ErrorCode> {
        if lambda <= 0.0 || !lambda.is_finite() {
            return Err(ErrorCode::InvalidInput);
        }
        let d = self.dim;
        let m = self.num_features;

        // Form Gram matrix G = U^T U + lambda * I_d [d, d]
        let mut g = vec![0.0_f64; d * d];
        for f in 0..m {
            let row = &self.directions[f * d..(f + 1) * d];
            for i in 0..d {
                let ri = row[i];
                let g_row = &mut g[i * d..(i + 1) * d];
                for j in i..d {
                    g_row[j] += ri * row[j];
                }
            }
        }
        for i in 0..d {
            for j in 0..i {
                g[i * d + j] = g[j * d + i];
            }
            g[i * d + i] += lambda;
        }

        // Gauss-Jordan inverse of symmetric PD matrix G [d, d]
        let mut inv = vec![0.0_f64; d * d];
        for i in 0..d {
            inv[i * d + i] = 1.0;
        }
        for col in 0..d {
            let mut pivot = col;
            let mut max_abs = g[col * d + col].abs();
            for r in (col + 1)..d {
                let v = g[r * d + col].abs();
                if v > max_abs {
                    max_abs = v;
                    pivot = r;
                }
            }
            if max_abs < 1e-14 {
                return Err(ErrorCode::SingularMatrix);
            }
            if pivot != col {
                for c in 0..d {
                    g.swap(col * d + c, pivot * d + c);
                    inv.swap(col * d + c, pivot * d + c);
                }
            }
            let inv_piv = 1.0 / g[col * d + col];
            for c in 0..d {
                g[col * d + c] *= inv_piv;
                inv[col * d + c] *= inv_piv;
            }
            for r in 0..d {
                if r == col {
                    continue;
                }
                let factor = g[r * d + col];
                if factor == 0.0 {
                    continue;
                }
                for c in 0..d {
                    g[r * d + c] -= factor * g[col * d + c];
                    inv[r * d + c] -= factor * inv[col * d + c];
                }
            }
        }

        // Compute oblique read vectors u_tilde_f = G^{-1} u_f, rescaled so dot(u_tilde_f, u_f) = 1.0
        let mut oblique = vec![0.0_f64; m * d];
        for f in 0..m {
            let u = &self.directions[f * d..(f + 1) * d];
            let dst = &mut oblique[f * d..(f + 1) * d];
            let mut self_dot = 0.0_f64;
            for r in 0..d {
                let inv_row = &inv[r * d..(r + 1) * d];
                let mut acc = 0.0_f64;
                for (&a, &b) in inv_row.iter().zip(u.iter()) {
                    acc += a * b;
                }
                dst[r] = acc;
                self_dot += acc * u[r];
            }
            if self_dot.abs() > 1e-12 {
                let scale = 1.0 / self_dot;
                for v in dst.iter_mut() {
                    *v *= scale;
                }
            }
        }

        Ok(FeatureSpace {
            num_features: m,
            dim: d,
            seed: self.seed,
            directions: oblique,
        })
    }
}

/// One literal in a boolean rule: `sign = +1` (feature present) or `-1` (NOT feature).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuleLiteral {
    pub feature: u32,
    pub sign: i8,
}

/// Symbolic rule specification before compilation into weights.
#[derive(Clone, Debug)]
pub struct RuleSpec {
    pub literals: Vec<RuleLiteral>,
    pub output: u32,
}

impl RuleSpec {
    /// Ground-truth boolean evaluation of this rule against a sorted or unsorted active feature set.
    #[inline]
    pub fn eval_ground_truth(&self, active_mask: &[bool]) -> bool {
        if self.literals.is_empty() {
            return false;
        }
        for lit in &self.literals {
            let is_on = active_mask
                .get(lit.feature as usize)
                .copied()
                .unwrap_or(false);
            if lit.sign > 0 && !is_on {
                return false;
            }
            if lit.sign < 0 && is_on {
                return false;
            }
        }
        true
    }
}

/// Preallocated hot-path scratch buffer for zero-allocation `RuleLayer` / stack execution.
#[derive(Clone, Debug)]
pub struct SuperpositionScratch {
    pub hidden_a: Vec<f64>,
    pub hidden_b: Vec<f64>,
    pub rule_acts: Vec<f64>,
}

impl SuperpositionScratch {
    pub fn new(max_dim: usize, max_rules: usize) -> Self {
        Self {
            hidden_a: vec![0.0_f64; max_dim],
            hidden_b: vec![0.0_f64; max_dim],
            rule_acts: vec![0.0_f64; max_rules],
        }
    }
}

/// Compiled superposition rule layer:
///   `a_r = max(w_r · h - b_r, 0) / (1 - tau)`
///   `h_out = sum_{r : a_r > 0} a_r * v_r`
#[derive(Clone, Debug)]
pub struct RuleLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_rules: usize,
    pub tau: f64,
    pub inv_one_minus_tau: f64,
    /// Row-major `[num_rules, in_dim]` read weights.
    pub w: Vec<f64>,
    /// Length `num_rules` biases: `(n_pos - 1) + tau`.
    pub b: Vec<f64>,
    /// Row-major `[num_rules, out_dim]` write directions (transposed `V` for sparse SAXPY).
    pub v_rows: Vec<f64>,
}

impl RuleLayer {
    pub fn compile(
        in_space: &FeatureSpace,
        out_space: &FeatureSpace,
        rules: &[RuleSpec],
        tau: f64,
        one_sentinel_feature: Option<u32>,
    ) -> Result<Self, ErrorCode> {
        if !(0.0..1.0).contains(&tau) || rules.is_empty() {
            return Err(ErrorCode::InvalidInput);
        }
        let in_dim = in_space.dim;
        let out_dim = out_space.dim;
        let num_rules = rules.len();

        let mut w = vec![0.0_f64; num_rules * in_dim];
        let mut b = vec![0.0_f64; num_rules];
        let mut v_rows = vec![0.0_f64; num_rules * out_dim];

        for (r, rule) in rules.iter().enumerate() {
            if rule.literals.is_empty() {
                return Err(ErrorCode::InvalidInput);
            }
            let w_row = &mut w[r * in_dim..(r + 1) * in_dim];
            let mut n_pos = 0usize;

            for lit in &rule.literals {
                let s = if lit.sign > 0 {
                    n_pos += 1;
                    1.0_f64
                } else if lit.sign < 0 {
                    -1.0_f64
                } else {
                    return Err(ErrorCode::InvalidInput);
                };
                let dir = in_space
                    .direction(lit.feature)
                    .ok_or(ErrorCode::InvalidInput)?;
                for (dst, &u) in w_row.iter_mut().zip(dir.iter()) {
                    *dst += s * u;
                }
            }

            // If all literals are negated, inject the always-on `__one__` sentinel feature
            if n_pos == 0 {
                let sentinel = one_sentinel_feature.ok_or(ErrorCode::InvalidInput)?;
                let dir = in_space
                    .direction(sentinel)
                    .ok_or(ErrorCode::InvalidInput)?;
                for (dst, &u) in w_row.iter_mut().zip(dir.iter()) {
                    *dst += u;
                }
                n_pos = 1;
            }

            b[r] = ((n_pos - 1) as f64) + tau;

            let v_dir = out_space
                .direction(rule.output)
                .ok_or(ErrorCode::InvalidInput)?;
            v_rows[r * out_dim..(r + 1) * out_dim].copy_from_slice(v_dir);
        }

        Ok(Self {
            in_dim,
            out_dim,
            num_rules,
            tau,
            inv_one_minus_tau: 1.0 / (1.0 - tau),
            w,
            b,
            v_rows,
        })
    }

    /// Compile a `RuleLayer` using oblique interference-cancelling read directions
    /// (`(U^T U + lambda I)^{-1} U^T`) while keeping `in_space` unit-norm for writes.
    pub fn compile_oblique(
        in_space: &FeatureSpace,
        out_space: &FeatureSpace,
        rules: &[RuleSpec],
        tau: f64,
        one_sentinel_feature: Option<u32>,
        whitening_lambda: f64,
    ) -> Result<Self, ErrorCode> {
        let oblique_in = in_space.compute_oblique_read_directions(whitening_lambda)?;
        Self::compile(&oblique_in, out_space, rules, tau, one_sentinel_feature)
    }

    /// Zero-allocation forward pass writing rule activations into `acts_out` and superposed output
    /// into `h_out`. Returns the number of active (non-zero) rule neurons.
    #[inline]
    pub fn forward_into(
        &self,
        h_in: &[f64],
        acts_out: &mut [f64],
        h_out: &mut [f64],
    ) -> Result<usize, ErrorCode> {
        if h_in.len() < self.in_dim || acts_out.len() < self.num_rules || h_out.len() < self.out_dim
        {
            return Err(ErrorCode::BufferTooSmall);
        }

        for v in h_out[..self.out_dim].iter_mut() {
            *v = 0.0;
        }

        let mut active_count = 0usize;
        for r in 0..self.num_rules {
            let w_row = &self.w[r * self.in_dim..(r + 1) * self.in_dim];
            let mut pre = -self.b[r];
            for (&wi, &hi) in w_row.iter().zip(h_in[..self.in_dim].iter()) {
                pre += wi * hi;
            }
            if pre > 0.0 {
                let a = pre * self.inv_one_minus_tau;
                acts_out[r] = a;
                active_count += 1;
                let v_row = &self.v_rows[r * self.out_dim..(r + 1) * self.out_dim];
                for (dst, &vi) in h_out[..self.out_dim].iter_mut().zip(v_row.iter()) {
                    *dst += a * vi;
                }
            } else {
                acts_out[r] = 0.0;
            }
        }
        Ok(active_count)
    }
}

/// Comparison direction for numeric threshold compilation.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThresholdOp {
    Greater = 0,
    Less = 1,
}

/// Numeric threshold specification: maps scalar input `x[input_idx]` to boolean feature `feature_idx`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NumericThresholdSpec {
    pub input_idx: u32,
    pub threshold: f64,
    pub steepness: f64,
    pub op: ThresholdOp,
    pub feature_idx: u32,
}

/// Compiled numeric threshold layer:
///   `f_k = clip(g_k * s_k * (x[input_idx] - t_k) + 0.5, 0, 1)`
///   `h += sum_k f_k * u_{feature_idx}`
#[derive(Clone, Debug)]
pub struct NumericThresholdLayer {
    pub num_inputs: usize,
    pub dim: usize,
    pub specs: Vec<NumericThresholdSpec>,
    /// Row-major `[specs.len(), dim]` feature directions (copied for contiguous hot-path SAXPY).
    pub write_dirs: Vec<f64>,
}

impl NumericThresholdLayer {
    pub fn compile(
        num_inputs: usize,
        space: &FeatureSpace,
        specs: &[NumericThresholdSpec],
    ) -> Result<Self, ErrorCode> {
        if num_inputs == 0 || specs.is_empty() {
            return Err(ErrorCode::InvalidInput);
        }
        let mut write_dirs = vec![0.0_f64; specs.len() * space.dim];
        for (i, sp) in specs.iter().enumerate() {
            if (sp.input_idx as usize) >= num_inputs
                || !sp.threshold.is_finite()
                || !sp.steepness.is_finite()
                || sp.steepness <= 0.0
            {
                return Err(ErrorCode::InvalidInput);
            }
            let dir = space
                .direction(sp.feature_idx)
                .ok_or(ErrorCode::InvalidInput)?;
            write_dirs[i * space.dim..(i + 1) * space.dim].copy_from_slice(dir);
        }
        Ok(Self {
            num_inputs,
            dim: space.dim,
            specs: specs.to_vec(),
            write_dirs,
        })
    }

    /// Evaluate thresholds on raw numeric `x` and accumulate into `h_out` (zero-allocation).
    #[inline]
    pub fn encode_into(&self, x: &[f64], h_out: &mut [f64], clear: bool) -> Result<(), ErrorCode> {
        if x.len() < self.num_inputs || h_out.len() < self.dim {
            return Err(ErrorCode::BufferTooSmall);
        }
        if clear {
            for v in h_out[..self.dim].iter_mut() {
                *v = 0.0;
            }
        }
        for (i, sp) in self.specs.iter().enumerate() {
            let val = x[sp.input_idx as usize];
            if !val.is_finite() {
                return Err(ErrorCode::InvalidInput);
            }
            let diff = match sp.op {
                ThresholdOp::Greater => val - sp.threshold,
                ThresholdOp::Less => sp.threshold - val,
            };
            let f = (sp.steepness * diff + 0.5).clamp(0.0, 1.0);
            if f > 0.0 {
                let dir = &self.write_dirs[i * self.dim..(i + 1) * self.dim];
                for (dst, &u) in h_out[..self.dim].iter_mut().zip(dir.iter()) {
                    *dst += f * u;
                }
            }
        }
        Ok(())
    }
}

/// Weighted linear scorer specification: `score = bias + sum_j coeff_j * a_{feature_j}`.
#[derive(Clone, Debug)]
pub struct LinearScorerSpec {
    pub terms: Vec<(u32, f64)>,
    pub bias: f64,
}

/// Compiled exact linear scorer layer operating directly on a superposed state `h in R^d`:
///   `y_k = w_k · h + bias_k`, where `w_k = sum_j c_{k,j} * u_j`.
#[derive(Clone, Debug)]
pub struct LinearScorerLayer {
    pub dim: usize,
    pub num_scores: usize,
    /// Row-major `[num_scores, dim]`.
    pub w: Vec<f64>,
    pub bias: Vec<f64>,
}

impl LinearScorerLayer {
    pub fn compile(space: &FeatureSpace, specs: &[LinearScorerSpec]) -> Result<Self, ErrorCode> {
        if specs.is_empty() {
            return Err(ErrorCode::InvalidInput);
        }
        let dim = space.dim;
        let num_scores = specs.len();
        let mut w = vec![0.0_f64; num_scores * dim];
        let mut bias = vec![0.0_f64; num_scores];

        for (k, sp) in specs.iter().enumerate() {
            if !sp.bias.is_finite() {
                return Err(ErrorCode::InvalidInput);
            }
            bias[k] = sp.bias;
            let row = &mut w[k * dim..(k + 1) * dim];
            for &(f_idx, coeff) in &sp.terms {
                if !coeff.is_finite() {
                    return Err(ErrorCode::InvalidInput);
                }
                let dir = space.direction(f_idx).ok_or(ErrorCode::InvalidInput)?;
                for (dst, &u) in row.iter_mut().zip(dir.iter()) {
                    *dst += coeff * u;
                }
            }
        }
        Ok(Self {
            dim,
            num_scores,
            w,
            bias,
        })
    }

    #[inline]
    pub fn forward_into(&self, h: &[f64], out: &mut [f64]) -> Result<(), ErrorCode> {
        if h.len() < self.dim || out.len() < self.num_scores {
            return Err(ErrorCode::BufferTooSmall);
        }
        for k in 0..self.num_scores {
            let row = &self.w[k * self.dim..(k + 1) * self.dim];
            let mut acc = self.bias[k];
            for (&wi, &hi) in row.iter().zip(h[..self.dim].iter()) {
                acc += wi * hi;
            }
            out[k] = acc;
        }
        Ok(())
    }
}

/// Binary decision tree node for compiling decision trees into superposition `RuleLayer` leaves.
#[derive(Clone, Debug)]
pub enum DecisionTreeNode {
    Split {
        feature: u32,
        when_true: Box<DecisionTreeNode>,
        when_false: Box<DecisionTreeNode>,
    },
    Leaf {
        output: u32,
    },
}

impl DecisionTreeNode {
    /// Lower every root-to-leaf path into an AND/NOT `RuleSpec`.
    pub fn compile_to_rules(&self, out_rules: &mut Vec<RuleSpec>) {
        let mut path = Vec::new();
        self.collect_paths(&mut path, out_rules);
    }

    fn collect_paths(&self, path: &mut Vec<RuleLiteral>, out_rules: &mut Vec<RuleSpec>) {
        match self {
            DecisionTreeNode::Leaf { output } => {
                if !path.is_empty() {
                    out_rules.push(RuleSpec {
                        literals: path.clone(),
                        output: *output,
                    });
                }
            }
            DecisionTreeNode::Split {
                feature,
                when_true,
                when_false,
            } => {
                path.push(RuleLiteral {
                    feature: *feature,
                    sign: 1,
                });
                when_true.collect_paths(path, out_rules);
                path.pop();

                path.push(RuleLiteral {
                    feature: *feature,
                    sign: -1,
                });
                when_false.collect_paths(path, out_rules);
                path.pop();
            }
        }
    }

    /// Ground-truth evaluation of the decision tree against an active boolean feature mask.
    pub fn eval_ground_truth(&self, active_mask: &[bool]) -> u32 {
        match self {
            DecisionTreeNode::Leaf { output } => *output,
            DecisionTreeNode::Split {
                feature,
                when_true,
                when_false,
            } => {
                let is_on = active_mask.get(*feature as usize).copied().unwrap_or(false);
                if is_on {
                    when_true.eval_ground_truth(active_mask)
                } else {
                    when_false.eval_ground_truth(active_mask)
                }
            }
        }
    }
}

/// Multi-layer chained superposition stack (`spaces[0] -> layers[0] -> spaces[1] -> ...`).
#[derive(Clone, Debug)]
pub struct SuperpositionStack {
    pub spaces: Vec<FeatureSpace>,
    pub layers: Vec<RuleLayer>,
}

impl SuperpositionStack {
    pub fn new(spaces: Vec<FeatureSpace>, layers: Vec<RuleLayer>) -> Result<Self, ErrorCode> {
        if spaces.len() != layers.len() + 1 {
            return Err(ErrorCode::InvalidArchitecture);
        }
        for (i, layer) in layers.iter().enumerate() {
            if layer.in_dim != spaces[i].dim || layer.out_dim != spaces[i + 1].dim {
                return Err(ErrorCode::ShapeMismatch);
            }
        }
        Ok(Self { spaces, layers })
    }

    /// Run all chained layers from `h_in` into `scratch`, returning the slice holding the final
    /// superposed state and total active rule neurons across all layers.
    pub fn forward_into<'a>(
        &self,
        h_in: &[f64],
        scratch: &'a mut SuperpositionScratch,
    ) -> Result<(&'a [f64], usize), ErrorCode> {
        let first_dim = self.spaces[0].dim;
        if h_in.len() < first_dim || scratch.hidden_a.len() < first_dim {
            return Err(ErrorCode::BufferTooSmall);
        }
        scratch.hidden_a[..first_dim].copy_from_slice(&h_in[..first_dim]);

        let mut total_active = 0usize;
        let mut in_a = true;
        for layer in &self.layers {
            if in_a {
                total_active += layer.forward_into(
                    &scratch.hidden_a[..layer.in_dim],
                    &mut scratch.rule_acts,
                    &mut scratch.hidden_b[..layer.out_dim],
                )?;
            } else {
                total_active += layer.forward_into(
                    &scratch.hidden_b[..layer.in_dim],
                    &mut scratch.rule_acts,
                    &mut scratch.hidden_a[..layer.out_dim],
                )?;
            }
            in_a = !in_a;
        }
        let out_dim = self.spaces.last().unwrap().dim;
        let out_slice = if in_a {
            &scratch.hidden_a[..out_dim]
        } else {
            &scratch.hidden_b[..out_dim]
        };
        Ok((out_slice, total_active))
    }
}
