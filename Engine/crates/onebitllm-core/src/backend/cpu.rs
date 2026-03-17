use ndarray::{Array, IxDyn, Ix2};

use crate::Result;
use crate::error::OneBitError;
use crate::tensor::PackedTensor;
use super::traits::ComputeBackend;

/// CPU compute backend using ndarray and bitpacked operations.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn packed_matmul(
        &self,
        packed: &PackedTensor,
        dense: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        packed.matmul_f32(dense)
    }

    fn dense_matmul(
        &self,
        a: &Array<f32, IxDyn>,
        b: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        let a2 = a
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
        let b2 = b
            .clone()
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
        Ok(a2.dot(&b2).into_dyn())
    }

    fn packed_matvec(
        &self,
        packed: &PackedTensor,
        input: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        packed.matvec(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::quant::QuantConfig;

    #[test]
    fn test_cpu_backend_packed_matmul() {
        let backend = CpuBackend;
        let lhs = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let rhs = array![[1.0f32, 0.0], [0.0, 1.0]].into_dyn();

        let packed = PackedTensor::from_ndarray(&lhs, &QuantConfig::per_tensor());
        let result = backend.packed_matmul(&packed, &rhs).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_cpu_backend_dense_matmul() {
        let backend = CpuBackend;
        let a = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
        let b = array![[5.0f32, 6.0], [7.0, 8.0]].into_dyn();

        let result = backend.dense_matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert!((result[[0, 0]] - 19.0).abs() < 1e-6);
        assert!((result[[0, 1]] - 22.0).abs() < 1e-6);
        assert!((result[[1, 0]] - 43.0).abs() < 1e-6);
        assert!((result[[1, 1]] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_backend_packed_matvec() {
        let backend = CpuBackend;
        let mat = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let vec = array![1.0f32, 2.0].into_dyn();

        let packed = PackedTensor::from_ndarray(&mat, &QuantConfig::per_tensor());
        let result = backend.packed_matvec(&packed, &vec).unwrap();
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn test_cpu_backend_name() {
        assert_eq!(CpuBackend.name(), "cpu");
    }
}
