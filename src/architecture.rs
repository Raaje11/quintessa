//! Tensor-level Architecture & Write-Permission Manifest (Milestone 1).
//!
//! Encodes component ownership, frozen/immutable invariants (VQ, SACI circuits),
//! bypassed paths (BrainDecoder), and maximum allowed patch delta norms.

use crate::checkpoint::{ParameterInventory, WeightPatch};
use crate::sys::{AdaComponent, ErrorCode, TensorId, WritePolicy};

#[derive(Clone, Copy, Debug)]
pub struct TensorPermission {
    pub tensor_id: TensorId,
    pub component: AdaComponent,
    pub declared_params: usize,
    pub policy: WritePolicy,
    pub max_delta_norm: f64,
    pub protected_subspace_rank: u16,
}

#[derive(Clone, Debug)]
pub struct ArchitectureManifest {
    pub permissions: Vec<TensorPermission>,
}

impl ArchitectureManifest {
    #[inline]
    pub fn permission(&self, id: TensorId) -> Option<&TensorPermission> {
        self.permissions.iter().find(|p| p.tensor_id == id)
    }

    pub fn total_declared_params(&self) -> usize {
        self.permissions.iter().map(|p| p.declared_params).sum()
    }

    /// Authoritative 7,395,228-parameter ADA-o1 component ownership & WSP policy manifest.
    pub fn ada_o1_manifest() -> Self {
        Self {
            permissions: vec![
                TensorPermission {
                    tensor_id: TensorId(0),
                    component: AdaComponent::BrainKda,
                    declared_params: 2_704_224,
                    policy: WritePolicy::EligibleTargeted,
                    max_delta_norm: 64.0,
                    protected_subspace_rank: 16,
                },
                TensorPermission {
                    tensor_id: TensorId(1),
                    component: AdaComponent::BrainMlaMoe,
                    declared_params: 819_080,
                    policy: WritePolicy::EligibleTargeted,
                    max_delta_norm: 64.0,
                    protected_subspace_rank: 16,
                },
                TensorPermission {
                    tensor_id: TensorId(2),
                    component: AdaComponent::BrainEmbedProj,
                    declared_params: 571_456,
                    policy: WritePolicy::EligibleWithInvariants,
                    max_delta_norm: 2.0,
                    protected_subspace_rank: 32,
                },
                TensorPermission {
                    tensor_id: TensorId(3),
                    component: AdaComponent::BrainDecoder,
                    declared_params: 306_867,
                    policy: WritePolicy::BypassedDoNotTarget,
                    max_delta_norm: 0.0,
                    protected_subspace_rank: 0,
                },
                TensorPermission {
                    tensor_id: TensorId(4),
                    component: AdaComponent::FrozenVq,
                    declared_params: 2_529_740,
                    policy: WritePolicy::Immutable,
                    max_delta_norm: 0.0,
                    protected_subspace_rank: 0,
                },
                TensorPermission {
                    tensor_id: TensorId(5),
                    component: AdaComponent::SaciCircuits,
                    declared_params: 22_886,
                    policy: WritePolicy::Immutable,
                    max_delta_norm: 0.0,
                    protected_subspace_rank: 0,
                },
                TensorPermission {
                    tensor_id: TensorId(6),
                    component: AdaComponent::SaciRouter,
                    declared_params: 87_119,
                    policy: WritePolicy::EligibleTargeted,
                    max_delta_norm: 32.0,
                    protected_subspace_rank: 8,
                },
                TensorPermission {
                    tensor_id: TensorId(7),
                    component: AdaComponent::WorkspaceProjNorm,
                    declared_params: 16_576,
                    policy: WritePolicy::EligibleTargeted,
                    max_delta_norm: 2.0,
                    protected_subspace_rank: 4,
                },
                TensorPermission {
                    tensor_id: TensorId(8),
                    component: AdaComponent::BridgesHaltingVerify,
                    declared_params: 154_239,
                    policy: WritePolicy::EligibleTargeted,
                    max_delta_norm: 4.0,
                    protected_subspace_rank: 8,
                },
                TensorPermission {
                    tensor_id: TensorId(9),
                    component: AdaComponent::ConceptGenerator,
                    declared_params: 183_041,
                    policy: WritePolicy::EligibleTargeted,
                    max_delta_norm: 5.0,
                    protected_subspace_rank: 8,
                },
            ],
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelGraph {
    pub writable_tensors: Vec<TensorId>,
    pub immutable_tensors: Vec<TensorId>,
    pub bypassed_tensors: Vec<TensorId>,
}

/// Reconstruct architectural write permissions against the inspected checkpoint inventory.
pub fn reconstruct_architecture(
    manifest: &ArchitectureManifest,
    inventory: &ParameterInventory,
) -> Result<ModelGraph, ErrorCode> {
    let mut writable_tensors = Vec::new();
    let mut immutable_tensors = Vec::new();
    let mut bypassed_tensors = Vec::new();

    for stat in &inventory.stats {
        let perm = manifest
            .permission(stat.id)
            .ok_or(ErrorCode::UnauthorizedWrite)?;
        if perm.component != stat.component {
            return Err(ErrorCode::ShapeMismatch);
        }
        match perm.policy {
            WritePolicy::Immutable => immutable_tensors.push(stat.id),
            WritePolicy::BypassedDoNotTarget => bypassed_tensors.push(stat.id),
            WritePolicy::EligibleWithInvariants | WritePolicy::EligibleTargeted => {
                writable_tensors.push(stat.id);
            }
        }
    }

    Ok(ModelGraph {
        writable_tensors,
        immutable_tensors,
        bypassed_tensors,
    })
}

/// Enforce write-permission and protected-invariant rules before applying patches.
pub fn validate_patch_permissions(
    patches: &[WeightPatch],
    manifest: &ArchitectureManifest,
) -> Result<(), ErrorCode> {
    for p in patches {
        let perm = manifest
            .permission(p.tensor_id)
            .ok_or(ErrorCode::UnauthorizedWrite)?;
        if !perm.policy.is_writable() {
            return Err(ErrorCode::ProtectedInvariant);
        }
        if p.delta_l2_norm() > perm.max_delta_norm {
            return Err(ErrorCode::ProtectedInvariant);
        }
    }
    Ok(())
}
