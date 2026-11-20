//! Solver Selection Engine & Native Rust Analytical/Derivative-Free Solvers (Milestone 3).
//!
//! No backpropagation, no external linear-algebra crates.

use crate::dcir::DomainOpKind;
use crate::sys::{AdaComponent, ErrorCode, SolverKind};

/// Select the weight-writing solver based on component ownership, operation,
/// and whether protected-subspace preservation is required.
pub fn select_solver(
    component: AdaComponent,
    op: DomainOpKind,
    has_protected_subspace: bool,
) -> Result<SolverKind, ErrorCode> {
    match component {
        AdaComponent::FrozenVq | AdaComponent::SaciCircuits | AdaComponent::BrainDecoder => {
            Err(ErrorCode::ProtectedInvariant)
        }
        AdaComponent::BridgesHaltingVerify => Ok(SolverKind::NelderMead),
        AdaComponent::BrainEmbedProj => Ok(SolverKind::SvdNullspace),
        AdaComponent::SaciRouter => match op {
            DomainOpKind::Division | DomainOpKind::Multiplication | DomainOpKind::Comparison => {
                Ok(SolverKind::ProgramToWeight)
            }
            _ => Ok(SolverKind::RidgePseudoinverse),
        },
        AdaComponent::BrainKda
        | AdaComponent::BrainMlaMoe
        | AdaComponent::WorkspaceProjNorm
        | AdaComponent::ConceptGenerator => {
            if has_protected_subspace {
                Ok(SolverKind::SvdNullspace)
            } else {
                match op {
                    DomainOpKind::InvariantCheck | DomainOpKind::Comparison => {
                        Ok(SolverKind::ProgramToWeight)
                    }
                    _ => Ok(SolverKind::RidgePseudoinverse),
                }
            }
        }
    }
}

/// Solve `A * x = b` in-place for `n x n` matrix `A` using Gaussian elimination with partial pivoting.
fn solve_linear_system(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<(), ErrorCode> {
    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = r;
            }
        }
        if max_val < 1e-12 {
            return Err(ErrorCode::SingularMatrix);
        }
        if pivot_row != col {
            for c in col..n {
                a.swap(col * n + c, pivot_row * n + c);
            }
            b.swap(col, pivot_row);
        }

        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / diag;
            a[r * n + col] = 0.0;
            for c in (col + 1)..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
            b[r] -= factor * b[col];
        }
    }

    for col in (0..n).rev() {
        let mut sum = b[col];
        for c in (col + 1)..n {
            sum -= a[col * n + c] * b[c];
        }
        b[col] = sum / a[col * n + col];
    }

    Ok(())
}

/// Analytical Ridge Regression / Tikhonov projection:
/// `W* = argmin_W ||X W - Y||_F^2 + lambda ||W - W0||_F^2`
/// Solved via `(X^T X + lambda I) W* = X^T Y + lambda W0`.
///
/// - `x`: `[n_samples, d_in]` row-major
/// - `y`: `[n_samples, d_out]` row-major
/// - `w0`: `[d_in, d_out]` row-major
pub fn solve_ridge_projection(
    n_samples: usize,
    d_in: usize,
    d_out: usize,
    x: &[f64],
    y: &[f64],
    w0: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, ErrorCode> {
    if n_samples == 0
        || d_in == 0
        || d_out == 0
        || x.len() != n_samples * d_in
        || y.len() != n_samples * d_out
        || w0.len() != d_in * d_out
        || lambda <= 0.0
    {
        return Err(ErrorCode::InvalidInput);
    }

    // Build X^T X + lambda * I  [d_in, d_in]
    let mut xtx = vec![0.0_f64; d_in * d_in];
    for s in 0..n_samples {
        let xs = &x[s * d_in..(s + 1) * d_in];
        for i in 0..d_in {
            for j in 0..d_in {
                xtx[i * d_in + j] += xs[i] * xs[j];
            }
        }
    }
    for i in 0..d_in {
        xtx[i * d_in + i] += lambda;
    }

    let mut w_star = vec![0.0_f64; d_in * d_out];
    let mut a_copy = vec![0.0_f64; d_in * d_in];
    let mut rhs = vec![0.0_f64; d_in];

    for col in 0..d_out {
        a_copy.copy_from_slice(&xtx);
        for i in 0..d_in {
            let mut acc = lambda * w0[i * d_out + col];
            for s in 0..n_samples {
                acc += x[s * d_in + i] * y[s * d_out + col];
            }
            rhs[i] = acc;
        }
        solve_linear_system(d_in, &mut a_copy, &mut rhs)?;
        for i in 0..d_in {
            w_star[i * d_out + col] = rhs[i];
        }
    }

    Ok(w_star)
}

/// Null-space constrained surgical editing:
/// Finds `dW` minimizing `||X_new (W0 + dW) - Y_target||_F^2 + lambda ||dW||_F^2`
/// subject to `X_protected * dW = 0` (exact preservation of protected activations).
pub fn solve_nullspace_edit(
    d_in: usize,
    d_out: usize,
    w0: &[f64],
    x_new: &[f64],
    y_target: &[f64],
    x_protected: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, ErrorCode> {
    if d_in == 0 || d_out == 0 || w0.len() != d_in * d_out || lambda <= 0.0 {
        return Err(ErrorCode::InvalidInput);
    }
    let n_new = x_new.len() / d_in;
    let n_prot = x_protected.len() / d_in;
    if n_new == 0 || x_new.len() != n_new * d_in || y_target.len() != n_new * d_out {
        return Err(ErrorCode::ShapeMismatch);
    }

    // 1. Orthonormalize rows of X_protected via modified Gram-Schmidt -> basis Q [k, d_in]
    let mut basis_q: Vec<Vec<f64>> = Vec::with_capacity(n_prot.min(d_in));
    for r in 0..n_prot {
        let mut v = x_protected[r * d_in..(r + 1) * d_in].to_vec();
        for q in &basis_q {
            let dot: f64 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            for i in 0..d_in {
                v[i] -= dot * q[i];
            }
        }
        let norm: f64 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for a in &mut v {
                *a /= norm;
            }
            basis_q.push(v);
        }
    }

    // 2. Project X_new into the orthogonal complement of span(X_protected): X_perp = X_new * (I - Q^T Q)
    let mut x_perp = x_new.to_vec();
    for s in 0..n_new {
        let row = &mut x_perp[s * d_in..(s + 1) * d_in];
        for q in &basis_q {
            let dot: f64 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            for i in 0..d_in {
                row[i] -= dot * q[i];
            }
        }
    }

    // 3. Residual target R = Y_target - X_new * W0
    let mut residual = vec![0.0_f64; n_new * d_out];
    for s in 0..n_new {
        for c in 0..d_out {
            let mut pred = 0.0;
            for i in 0..d_in {
                pred += x_new[s * d_in + i] * w0[i * d_out + c];
            }
            residual[s * d_out + c] = y_target[s * d_out + c] - pred;
        }
    }

    // 4. Solve ridge for dW_raw on X_perp -> R, with zero prior (W0 = 0)
    let zero_w0 = vec![0.0_f64; d_in * d_out];
    let mut dw = solve_ridge_projection(n_new, d_in, d_out, &x_perp, &residual, &zero_w0, lambda)?;

    // 5. Explicitly project columns of dW onto null-space (I - Q^T Q) to guarantee X_protected * dW == 0
    for c in 0..d_out {
        for q in &basis_q {
            let mut dot = 0.0;
            for i in 0..d_in {
                dot += q[i] * dw[i * d_out + c];
            }
            for i in 0..d_in {
                dw[i * d_out + c] -= dot * q[i];
            }
        }
    }

    Ok(dw)
}

/// Derivative-free grid + golden-section calibration of `halt_threshold` for Adaptive PonderNet.
pub fn calibrate_ponder_threshold<F>(mut eval_loss: F, low: f64, high: f64, iters: usize) -> f64
where
    F: FnMut(f64) -> f64,
{
    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut a = low;
    let mut b = high;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = eval_loss(c);
    let mut fd = eval_loss(d);

    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = eval_loss(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = eval_loss(d);
        }
    }

    0.5 * (a + b)
}
