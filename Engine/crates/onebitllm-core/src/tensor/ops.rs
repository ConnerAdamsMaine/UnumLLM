use ndarray::{Array, Ix2, IxDyn};

use super::packed_tensor::PackedTensor;
use crate::error::OneBitError;
use crate::Result;

/// Dense f32 matrix * packed ternary matrix.
///
/// `dense` is (M x K) f32, `packed` is (K x N) quantized.
/// Returns (M x N) f32.
pub fn matmul_dense_packed(
    dense: &Array<f32, IxDyn>,
    packed: &PackedTensor,
) -> Result<Array<f32, IxDyn>> {
    if dense.ndim() != 2 || packed.ndim() != 2 {
        return Err(OneBitError::TensorOp(
            "matmul_dense_packed requires 2D tensors".into(),
        ));
    }

    let k1 = dense.shape()[1];
    let k2 = packed.shape()[0];

    if k1 != k2 {
        return Err(OneBitError::ShapeMismatch {
            expected: vec![k1],
            got: vec![k2],
        });
    }

    // Dequantize packed, then use ndarray matmul
    let rhs = packed.to_ndarray();
    let lhs2 = dense
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
    let rhs2 = rhs
        .into_dimensionality::<Ix2>()
        .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

    Ok(lhs2.dot(&rhs2).into_dyn())
}

/// Element-wise add of two packed tensors (returns dense f32).
///
/// Dequantizes both operands and adds them.
pub fn packed_add(a: &PackedTensor, b: &PackedTensor) -> Result<Array<f32, IxDyn>> {
    if a.shape() != b.shape() {
        return Err(OneBitError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let a_dense = a.to_ndarray();
    let b_dense = b.to_ndarray();
    Ok(&a_dense + &b_dense)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantConfig;
    use ndarray::array;

    #[test]
    fn test_matmul_dense_packed() {
        let dense = array![[1.0f32, 0.0], [0.0, 1.0]].into_dyn();
        let packed_arr = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&packed_arr, &QuantConfig::per_tensor());

        let result = matmul_dense_packed(&dense, &packed).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_packed_add() {
        let arr_a = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let arr_b = array![[1.0f32, 1.0], [-1.0, 0.0]].into_dyn();

        let packed_a = PackedTensor::from_ndarray(&arr_a, &QuantConfig::per_tensor());
        let packed_b = PackedTensor::from_ndarray(&arr_b, &QuantConfig::per_tensor());

        let result = packed_add(&packed_a, &packed_b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_packed_add_shape_mismatch() {
        let arr_a = Array::from_elem(IxDyn(&[2, 3]), 1.0f32);
        let arr_b = Array::from_elem(IxDyn(&[3, 2]), 1.0f32);

        let packed_a = PackedTensor::from_ndarray(&arr_a, &QuantConfig::per_tensor());
        let packed_b = PackedTensor::from_ndarray(&arr_b, &QuantConfig::per_tensor());

        assert!(packed_add(&packed_a, &packed_b).is_err());
    }
}
