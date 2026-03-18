use ndarray::{Array, IxDyn};

use crate::Result;
use crate::error::OneBitError;
use crate::tensor::PackedTensor;
use super::traits::ComputeBackend;

/// ROCm backend scaffold.
///
/// This exists to keep device plumbing and build surfaces stable while HIP
/// kernels are wired in. The current implementation still fails explicitly at
/// execution time instead of silently falling back.
pub struct RocmBackend;

impl RocmBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn packed_matmul(
        &self,
        _packed: &PackedTensor,
        _dense: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        Err(OneBitError::Other(
            "ROCm backend scaffold is enabled, but packed HIP kernels are not implemented yet".into(),
        ))
    }

    fn dense_matmul(
        &self,
        _a: &Array<f32, IxDyn>,
        _b: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        Err(OneBitError::Other(
            "ROCm backend scaffold is enabled, but dense HIP kernels are not implemented yet".into(),
        ))
    }

    fn packed_matmul_dense_left_transposed(
        &self,
        _lhs: &Array<f32, IxDyn>,
        _packed: &PackedTensor,
    ) -> Result<Array<f32, IxDyn>> {
        Err(OneBitError::Other(
            "ROCm backend scaffold is enabled, but dense-left packed HIP kernels are not implemented yet".into(),
        ))
    }

    fn packed_matvec(
        &self,
        _packed: &PackedTensor,
        _input: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        Err(OneBitError::Other(
            "ROCm backend scaffold is enabled, but packed matvec HIP kernels are not implemented yet".into(),
        ))
    }
}
