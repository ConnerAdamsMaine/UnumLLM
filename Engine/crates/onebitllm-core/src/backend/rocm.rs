use ndarray::{Array, IxDyn};

use super::hip_runtime;
use super::traits::ComputeBackend;
use crate::error::OneBitError;
use crate::tensor::PackedTensor;
use crate::Result;

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

    fn preflight_packed_2d(&self, packed: &PackedTensor) -> Result<String> {
        let layout = packed.kernel_layout_2d()?;
        Ok(layout.describe())
    }

    fn preflight_dense_2d(&self, name: &str, value: &Array<f32, IxDyn>) -> Result<(usize, usize)> {
        if value.ndim() != 2 {
            return Err(OneBitError::TensorOp(format!(
                "{name} must be 2D for ROCm kernel launch, got {}D",
                value.ndim()
            )));
        }
        Ok((value.shape()[0], value.shape()[1]))
    }
}

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn packed_matmul(
        &self,
        packed: &PackedTensor,
        dense: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        let packed_summary = self.preflight_packed_2d(packed)?;
        let (dense_rows, dense_cols) = self.preflight_dense_2d("dense rhs", dense)?;
        Err(OneBitError::Other(
            format!(
                "ROCm backend scaffold is enabled, but packed HIP kernels are not implemented yet (prepared packed={} rhs={}x{})",
                packed_summary,
                dense_rows,
                dense_cols
            ),
        ))
    }

    fn dense_matmul(
        &self,
        a: &Array<f32, IxDyn>,
        b: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        let (a_rows, a_cols) = self.preflight_dense_2d("dense lhs", a)?;
        let (b_rows, b_cols) = self.preflight_dense_2d("dense rhs", b)?;
        Err(OneBitError::Other(
            format!(
                "ROCm backend scaffold is enabled, but dense HIP kernels are not implemented yet (lhs={}x{} rhs={}x{})",
                a_rows,
                a_cols,
                b_rows,
                b_cols
            ),
        ))
    }

    fn packed_matmul_dense_left_transposed(
        &self,
        lhs: &Array<f32, IxDyn>,
        packed: &PackedTensor,
    ) -> Result<Array<f32, IxDyn>> {
        let _ = self.preflight_dense_2d("dense lhs", lhs)?;
        let _ = self.preflight_packed_2d(packed)?;
        hip_runtime::packed_matmul_dense_left_transposed(packed, lhs)
    }

    fn packed_matvec(
        &self,
        packed: &PackedTensor,
        input: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        let packed_summary = self.preflight_packed_2d(packed)?;
        if input.ndim() != 1 {
            return Err(OneBitError::TensorOp(format!(
                "dense input must be 1D for ROCm packed matvec launch, got {}D",
                input.ndim()
            )));
        }
        Err(OneBitError::Other(
            format!(
                "ROCm backend scaffold is enabled, but packed matvec HIP kernels are not implemented yet (packed={} input_len={})",
                packed_summary,
                input.len()
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{QuantConfig, QuantGranularity};
    use ndarray::array;

    #[test]
    fn test_rocm_backend_dense_left_reports_missing_or_runtime_kernel_state() {
        let backend = RocmBackend::new();
        let packed = PackedTensor::from_binary_ndarray(
            &array![[1.0f32, -1.0], [0.5, -0.5]].into_dyn(),
            QuantGranularity::PerTensor,
            7,
        );
        let lhs = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();

        let err = backend
            .packed_matmul_dense_left_transposed(&lhs, &packed)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no HIP kernel library was configured")
                || err.contains("HIP kernel library path does not exist")
                || err.contains("HIP packed dense-left matmul entrypoint returned error code")
                || err.contains("failed to load HIP kernel library")
        );
    }

    #[test]
    fn test_rocm_backend_preflight_rejects_non_matrix_packed_layout() {
        let backend = RocmBackend::new();
        let packed = PackedTensor::from_ndarray(
            &ndarray::Array::from_elem(IxDyn(&[2, 2, 2]), 1.0f32),
            &QuantConfig::per_tensor(),
        );
        let rhs = array![[1.0f32], [2.0f32]].into_dyn();

        let err = backend
            .packed_matmul(&packed, &rhs)
            .unwrap_err()
            .to_string();
        assert!(err.contains("kernel_layout_2d requires a 2D packed tensor"));
    }
}
