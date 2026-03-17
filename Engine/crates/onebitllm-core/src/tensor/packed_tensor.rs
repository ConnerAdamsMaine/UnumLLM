use ndarray::{Array, Array2, IxDyn, Ix2};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::Result;
use crate::error::OneBitError;
use crate::quant::{PackedTernary, QuantConfig, QuantGranularity, QuantParams, TernaryWeight};
use super::shape;

/// A quantized tensor: stores ternary weights in bitpacked form alongside
/// quantization parameters. Conceptually represents an ndarray of {-1, 0, +1}
/// with associated scale factors.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    /// Packed ternary data (flat, row-major).
    packed: PackedTernary,
    /// Shape of the logical tensor.
    shape: Vec<usize>,
    /// Quantization parameters (scales, zero-points).
    quant_params: QuantParams,
}

impl PackedTensor {
    /// Create from an f32 ndarray by quantizing.
    pub fn from_ndarray(arr: &Array<f32, IxDyn>, config: &QuantConfig) -> Self {
        let shape = arr.shape().to_vec();
        let flat: Vec<f32> = arr.iter().copied().collect();
        let quant_params = QuantParams::compute(&flat, &shape, config);

        // Quantize each group using its computed scale
        let mut ternary = Vec::with_capacity(flat.len());
        match config.granularity {
            crate::quant::QuantGranularity::PerTensor => {
                let scale = quant_params.scales[0];
                for &w in &flat {
                    ternary.push(TernaryWeight::quantize(w, scale));
                }
            }
            crate::quant::QuantGranularity::PerChannel => {
                let channel_size = flat.len() / shape[0];
                for (c, chunk) in flat.chunks(channel_size).enumerate() {
                    let scale = quant_params.scales[c];
                    for &w in chunk {
                        ternary.push(TernaryWeight::quantize(w, scale));
                    }
                }
            }
            crate::quant::QuantGranularity::PerGroup(group_size) => {
                for (g, chunk) in flat.chunks(group_size).enumerate() {
                    let scale = quant_params.scales[g];
                    for &w in chunk {
                        ternary.push(TernaryWeight::quantize(w, scale));
                    }
                }
            }
        }

        let packed = PackedTernary::from_ternary_slice(&ternary);

        Self {
            packed,
            shape,
            quant_params,
        }
    }

    /// Create from a 2D f32 array (most common: weight matrices).
    pub fn from_array2(arr: &Array2<f32>, config: &QuantConfig) -> Self {
        Self::from_ndarray(&arr.clone().into_dyn(), config)
    }

    /// Dequantize back to f32 ndarray.
    ///
    /// Each ternary weight is multiplied by its group's scale factor.
    pub fn to_ndarray(&self) -> Array<f32, IxDyn> {
        let total = shape::num_elements(&self.shape);
        let mut flat = Vec::with_capacity(total);

        for i in 0..total {
            let w = self.packed.get(i);
            let scale = self.quant_params.scale_for_index(i);
            flat.push(w.to_f32() * scale);
        }

        Array::from_shape_vec(IxDyn(&self.shape), flat)
            .expect("shape mismatch in dequantization")
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        shape::num_elements(&self.shape)
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reshape (repacks into new logical shape, same data).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        shape::validate_reshape(&self.shape, new_shape)?;
        Ok(Self {
            packed: self.packed.clone(),
            shape: new_shape.to_vec(),
            quant_params: QuantParams {
                original_shape: new_shape.to_vec(),
                ..self.quant_params.clone()
            },
        })
    }

    /// Slice along a dimension (returns a new packed tensor for the slice).
    ///
    /// For example, for a (M, N) matrix, `slice(0, 2, 5)` extracts rows 2..5.
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Self> {
        if dim >= self.shape.len() {
            return Err(OneBitError::TensorOp(format!(
                "Dimension {dim} out of bounds for {}-d tensor",
                self.shape.len()
            )));
        }
        if end > self.shape[dim] || start >= end {
            return Err(OneBitError::TensorOp(format!(
                "Invalid slice [{start}..{end}) for dimension {dim} of size {}",
                self.shape[dim]
            )));
        }

        // Dequantize, slice in ndarray, re-quantize
        let arr = self.to_ndarray();
        let sliced = arr.slice_axis(
            ndarray::Axis(dim),
            ndarray::Slice::from(start..end),
        );
        let owned = sliced.to_owned();

        // Rebuild with per-tensor quantization for simplicity
        let config = QuantConfig::per_tensor();
        Ok(Self::from_ndarray(&owned, &config))
    }

    /// Transpose a 2D tensor.
    pub fn t(&self) -> Result<Self> {
        if self.shape.len() != 2 {
            return Err(OneBitError::TensorOp(format!(
                "Transpose requires 2D tensor, got {}D",
                self.shape.len()
            )));
        }

        let arr = self.to_ndarray();
        let arr2 = arr
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
        let transposed = arr2.t().to_owned();

        let config = QuantConfig {
            granularity: self.quant_params.granularity,
            ..QuantConfig::default()
        };
        Ok(Self::from_array2(&transposed, &config))
    }

    /// Matrix-vector product: self (M x N packed) * input (N,) f32 -> output (M,) f32.
    ///
    /// Uses bitpacked dot product for each row.
    pub fn matvec(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        if self.shape.len() != 2 {
            return Err(OneBitError::TensorOp(
                "matvec requires 2D packed tensor".into(),
            ));
        }

        let m = self.shape[0];
        let n = self.shape[1];
        let input_flat: Vec<f32> = input.iter().copied().collect();

        if input_flat.len() != n {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![n],
                got: vec![input_flat.len()],
            });
        }

        #[cfg(feature = "rayon")]
        let output: Vec<f32> = (0..m)
            .into_par_iter()
            .map(|row| self.row_dot(row, &input_flat))
            .collect();

        #[cfg(not(feature = "rayon"))]
        let output: Vec<f32> = (0..m)
            .map(|row| self.row_dot(row, &input_flat))
            .collect();

        Ok(Array::from_shape_vec(IxDyn(&[m]), output).unwrap())
    }

    /// Matrix-matrix product: self (M x K packed) * rhs (K x N f32) -> output (M x N f32).
    pub fn matmul_f32(&self, rhs: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        if self.shape.len() != 2 {
            return Err(OneBitError::TensorOp(
                "matmul requires 2D packed tensor".into(),
            ));
        }
        if rhs.ndim() != 2 {
            return Err(OneBitError::TensorOp(
                "matmul requires 2D right-hand operand".into(),
            ));
        }

        let k = self.shape[1];
        let rhs_shape = rhs.shape();
        let k2 = rhs_shape[0];

        if k != k2 {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![k],
                got: vec![k2],
            });
        }

        let rhs2 = rhs
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
        let m = self.shape[0];
        let n = rhs2.shape()[1];
        let rhs_columns: Vec<Vec<f32>> = (0..n)
            .map(|col| rhs2.column(col).iter().copied().collect())
            .collect();

        #[cfg(feature = "rayon")]
        let rows: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|row| rhs_columns.iter().map(|col| self.row_dot(row, col)).collect())
            .collect();

        #[cfg(not(feature = "rayon"))]
        let rows: Vec<Vec<f32>> = (0..m)
            .map(|row| rhs_columns.iter().map(|col| self.row_dot(row, col)).collect())
            .collect();

        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let result = Array2::from_shape_vec((m, n), flat)
            .expect("packed matmul output shape should be valid");
        Ok(result.into_dyn())
    }

    /// Dense matrix (B x K) * packed matrix^T (M x K)^T -> dense (B x M).
    ///
    /// This is the hot inference path for quantized linear layers: inputs stay
    /// dense and weights stay packed for the whole multiply.
    pub fn matmul_dense_left_transposed(
        &self,
        lhs: &Array<f32, IxDyn>,
    ) -> Result<Array<f32, IxDyn>> {
        if self.shape.len() != 2 {
            return Err(OneBitError::TensorOp(
                "matmul_dense_left_transposed requires 2D packed tensor".into(),
            ));
        }
        if lhs.ndim() != 2 {
            return Err(OneBitError::TensorOp(
                "matmul_dense_left_transposed requires 2D left-hand operand".into(),
            ));
        }

        let lhs2 = lhs
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
        let in_features = self.shape[1];
        if lhs2.shape()[1] != in_features {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![in_features],
                got: vec![lhs2.shape()[1]],
            });
        }

        let batch = lhs2.shape()[0];
        let out_features = self.shape[0];
        let lhs_rows: Vec<Vec<f32>> = (0..batch)
            .map(|batch_idx| lhs2.row(batch_idx).iter().copied().collect())
            .collect();

        #[cfg(feature = "rayon")]
        let rows: Vec<Vec<f32>> = lhs_rows
            .par_iter()
            .map(|input_row| {
                (0..out_features)
                    .map(|out_idx| self.row_dot(out_idx, input_row))
                    .collect()
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        let rows: Vec<Vec<f32>> = lhs_rows
            .iter()
            .map(|input_row| {
                (0..out_features)
                    .map(|out_idx| self.row_dot(out_idx, input_row))
                    .collect()
            })
            .collect();

        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let result = Array2::from_shape_vec((batch, out_features), flat)
            .expect("packed dense-left matmul output shape should be valid");
        Ok(result.into_dyn())
    }

    /// Access raw packed data.
    pub fn packed_data(&self) -> &PackedTernary {
        &self.packed
    }

    /// Access quantization params.
    pub fn quant_params(&self) -> &QuantParams {
        &self.quant_params
    }

    /// Memory usage in bytes (packed data only).
    pub fn memory_bytes(&self) -> usize {
        self.packed.memory_bytes()
    }

    fn row_dot(&self, row: usize, input: &[f32]) -> f32 {
        let n = self.shape[1];
        let row_start = row * n;

        match self.quant_params.granularity {
            QuantGranularity::PerTensor => {
                self.packed
                    .dot_slice_f32(row_start, input, self.quant_params.scales[0])
                    .expect("validated packed row slice should fit input")
            }
            QuantGranularity::PerChannel if self.shape.len() == 2 => {
                self.packed
                    .dot_slice_f32(row_start, input, self.quant_params.scales[row])
                    .expect("validated packed row slice should fit input")
            }
            QuantGranularity::PerGroup(_) => {
                let mut acc = 0.0f32;
                for (col, &value) in input.iter().enumerate() {
                    let flat_idx = row_start + col;
                    let w = self.packed.get(flat_idx);
                    let scale = self.quant_params.scale_for_index(flat_idx);
                    acc += w.to_f32() * scale * value;
                }
                acc
            }
            QuantGranularity::PerChannel => {
                let mut acc = 0.0f32;
                for (col, &value) in input.iter().enumerate() {
                    let flat_idx = row_start + col;
                    let w = self.packed.get(flat_idx);
                    let scale = self.quant_params.scale_for_index(flat_idx);
                    acc += w.to_f32() * scale * value;
                }
                acc
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_from_ndarray_roundtrip() {
        // Values exactly at {-1, 0, +1} should survive per-tensor quantization
        let arr = array![[1.0f32, -1.0, 0.0], [0.0, 1.0, -1.0]].into_dyn();
        let config = QuantConfig::per_tensor();
        let packed = PackedTensor::from_ndarray(&arr, &config);

        assert_eq!(packed.shape(), &[2, 3]);
        assert_eq!(packed.len(), 6);

        let dequant = packed.to_ndarray();
        assert_eq!(dequant.shape(), &[2, 3]);

        // Dequantized values should be scale * {-1, 0, +1}
        // With absmean scale = (1+1+0+0+1+1)/6 = 4/6 = 0.667
        let scale = packed.quant_params().scales[0];
        for (orig, deq) in arr.iter().zip(dequant.iter()) {
            if *orig > 0.0 {
                assert!((*deq - scale).abs() < 1e-6);
            } else if *orig < 0.0 {
                assert!((*deq + scale).abs() < 1e-6);
            } else {
                assert!(deq.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_from_array2() {
        let arr = array![[1.0f32, -1.0], [0.0, 1.0]];
        let config = QuantConfig::per_tensor();
        let packed = PackedTensor::from_array2(&arr, &config);
        assert_eq!(packed.shape(), &[2, 2]);
    }

    #[test]
    fn test_reshape() {
        let arr = array![[1.0f32, -1.0, 0.0], [0.0, 1.0, -1.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());

        let reshaped = packed.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.len(), 6);
    }

    #[test]
    fn test_reshape_invalid() {
        let arr = Array::from_elem(IxDyn(&[2, 3]), 1.0f32);
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());
        assert!(packed.reshape(&[5]).is_err());
    }

    #[test]
    fn test_slice() {
        let arr = array![[1.0f32, -1.0], [0.0, 1.0], [-1.0, 0.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());

        // Slice rows 0..2
        let sliced = packed.slice(0, 0, 2).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);
    }

    #[test]
    fn test_transpose() {
        let arr = array![[1.0f32, -1.0, 0.0], [0.0, 1.0, -1.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());

        let transposed = packed.t().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
    }

    #[test]
    fn test_matvec() {
        // 2x3 matrix * 3-vector
        let arr = array![[1.0f32, -1.0, 0.0], [0.0, 1.0, -1.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());
        let input = array![1.0f32, 2.0, 3.0].into_dyn();

        let result = packed.matvec(&input).unwrap();
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn test_matmul_f32() {
        // 2x3 packed * 3x2 dense
        let lhs = array![[1.0f32, -1.0, 0.0], [0.0, 1.0, -1.0]].into_dyn();
        let rhs = array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn();

        let packed = PackedTensor::from_ndarray(&lhs, &QuantConfig::per_tensor());
        let result = packed.matmul_f32(&rhs).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_matmul_dense_left_transposed() {
        let packed_weights = array![[1.0f32, -1.0, 0.0], [0.0, 1.0, 1.0]].into_dyn();
        let lhs = array![[2.0f32, 3.0, 4.0], [1.0, 0.0, 5.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&packed_weights, &QuantConfig::per_channel());

        let result = packed.matmul_dense_left_transposed(&lhs).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let lhs = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let rhs = array![[1.0f32, 0.0, 0.0]].into_dyn();

        let packed = PackedTensor::from_ndarray(&lhs, &QuantConfig::per_tensor());
        assert!(packed.matmul_f32(&rhs).is_err());
    }

    #[test]
    fn test_per_channel_quantization() {
        let arr = array![
            [10.0f32, -10.0, 0.0],
            [0.1, -0.1, 0.0]
        ]
        .into_dyn();
        let config = QuantConfig::per_channel();
        let packed = PackedTensor::from_ndarray(&arr, &config);

        assert_eq!(packed.quant_params().scales.len(), 2);
        // Channel 0 should have much larger scale than channel 1
        assert!(packed.quant_params().scales[0] > packed.quant_params().scales[1]);
    }

    #[test]
    fn test_memory_efficiency() {
        // 1024 weights: packed = 1024 * 2 bits / 8 = 256 bytes
        // vs f32 = 1024 * 4 = 4096 bytes (16x compression)
        let arr = Array::from_elem(IxDyn(&[32, 32]), 1.0f32);
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());
        let packed_bytes = packed.memory_bytes();
        let f32_bytes = 32 * 32 * 4;
        assert!(packed_bytes * 16 <= f32_bytes);
    }
}
