//! Checkpoint inspection and sparse patch application.

use crate::sys::{AdaComponent, ErrorCode, TensorId};

#[derive(Clone, Copy, Debug)]
pub struct TensorDescriptor {
    pub id: TensorId,
    pub component: AdaComponent,
    pub rows: usize,
    pub cols: usize,
    pub offset: usize,
    pub len: usize,
    pub frozen: bool,
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub descriptors: Vec<TensorDescriptor>,
    pub data: Vec<f64>,
}

impl Checkpoint {
    pub fn new(descriptors: Vec<TensorDescriptor>, data: Vec<f64>) -> Result<Self, ErrorCode> {
        for d in &descriptors {
            if d.rows.checked_mul(d.cols) != Some(d.len) {
                return Err(ErrorCode::ShapeMismatch);
            }
            if d.offset
                .checked_add(d.len)
                .is_none_or(|end| end > data.len())
            {
                return Err(ErrorCode::ShapeMismatch);
            }
        }
        Ok(Self { descriptors, data })
    }

    #[inline]
    pub fn descriptor(&self, id: TensorId) -> Option<&TensorDescriptor> {
        self.descriptors.iter().find(|d| d.id == id)
    }

    #[inline]
    pub fn tensor_slice(&self, id: TensorId) -> Option<&[f64]> {
        let d = self.descriptor(id)?;
        Some(&self.data[d.offset..d.offset + d.len])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TensorStats {
    pub id: TensorId,
    pub component: AdaComponent,
    pub len: usize,
    pub nonzero_count: usize,
    pub min_val: f64,
    pub max_val: f64,
    pub l2_norm_sq: f64,
}

#[derive(Clone, Debug)]
pub struct ParameterInventory {
    pub total_params: usize,
    pub trainable_params: usize,
    pub frozen_params: usize,
    pub stats: Vec<TensorStats>,
}

/// Inspect original model weights, numerical ranges, sparsity, and frozen counts.
pub fn inspect_checkpoint(ckpt: &Checkpoint) -> Result<ParameterInventory, ErrorCode> {
    let mut total_params = 0usize;
    let mut trainable_params = 0usize;
    let mut frozen_params = 0usize;
    let mut stats = Vec::with_capacity(ckpt.descriptors.len());

    for d in &ckpt.descriptors {
        let slice = &ckpt.data[d.offset..d.offset + d.len];
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut nonzero_count = 0usize;
        let mut l2_norm_sq = 0.0_f64;

        for &v in slice {
            if !v.is_finite() {
                return Err(ErrorCode::InvalidInput);
            }
            if v != 0.0 {
                nonzero_count += 1;
            }
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
            l2_norm_sq += v * v;
        }
        if slice.is_empty() {
            min_val = 0.0;
            max_val = 0.0;
        }

        total_params += d.len;
        if d.frozen {
            frozen_params += d.len;
        } else {
            trainable_params += d.len;
        }

        stats.push(TensorStats {
            id: d.id,
            component: d.component,
            len: d.len,
            nonzero_count,
            min_val,
            max_val,
            l2_norm_sq,
        });
    }

    Ok(ParameterInventory {
        total_params,
        trainable_params,
        frozen_params,
        stats,
    })
}

#[derive(Clone, Debug)]
pub struct WeightPatch {
    pub tensor_id: TensorId,
    /// Sparse `(flat_index, delta_value)` updates.
    pub sparse_deltas: Vec<(usize, f64)>,
}

impl WeightPatch {
    pub fn delta_l2_norm(&self) -> f64 {
        let sq: f64 = self.sparse_deltas.iter().map(|&(_, dv)| dv * dv).sum();
        sq.sqrt()
    }
}

/// Apply validated sparse weight patches to produce a candidate checkpoint.
pub fn apply_patches(ckpt: &Checkpoint, patches: &[WeightPatch]) -> Result<Checkpoint, ErrorCode> {
    let mut out = ckpt.clone();
    for p in patches {
        let desc = out
            .descriptor(p.tensor_id)
            .ok_or(ErrorCode::ShapeMismatch)?
            .to_owned();
        if desc.frozen {
            return Err(ErrorCode::UnauthorizedWrite);
        }
        for &(idx, dv) in &p.sparse_deltas {
            if idx >= desc.len || !dv.is_finite() {
                return Err(ErrorCode::ShapeMismatch);
            }
            out.data[desc.offset + idx] += dv;
        }
    }
    Ok(out)
}
