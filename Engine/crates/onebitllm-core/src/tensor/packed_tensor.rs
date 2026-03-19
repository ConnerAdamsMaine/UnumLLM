use ndarray::{Array, Array2, Ix2, IxDyn};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::shape;
use crate::error::OneBitError;
use crate::quant::{
    effective_sign_from_toggle, toggle_bit_for_sign, PackedBinary, PackedTernary, QuantConfig,
    QuantGranularity, QuantParams, TernaryWeight,
};
use crate::Result;

/// Packed runtime weight storage.
#[derive(Debug, Clone)]
pub enum PackedStorage {
    /// Packed ternary weights in {-1, 0, +1}.
    Ternary(PackedTernary),
    /// Packed binary toggle bits with metadata-driven sign reconstruction.
    Binary {
        data: PackedBinary,
        equalizer_seed: u64,
    },
}

/// Runtime weight format represented by a packed tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedWeightFormat {
    Ternary,
    Binary,
}

/// Scale layout presented to low-level packed-matmul kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedScaleLayout {
    /// One scale for the full tensor.
    PerTensor,
    /// One scale per output row/channel.
    PerRow,
    /// Consecutive flat groups share a scale.
    PerGroup { group_size: usize },
}

/// Borrowed 2D packed-weight view for backend kernels.
///
/// This keeps the Rust-side quantization layout explicit so device backends can
/// launch kernels without re-deriving row/scale/equalizer metadata ad hoc.
#[derive(Debug, Clone)]
pub struct PackedTensorKernelLayout<'a> {
    pub weight_format: PackedWeightFormat,
    pub scale_layout: PackedScaleLayout,
    pub rows: usize,
    pub cols: usize,
    pub bits_per_weight: u8,
    pub packed_words: &'a [u64],
    pub scales: &'a [f32],
    pub equalizer_seed: Option<u64>,
}

impl PackedTensorKernelLayout<'_> {
    pub fn describe(&self) -> String {
        let format = match self.weight_format {
            PackedWeightFormat::Ternary => "ternary",
            PackedWeightFormat::Binary => "binary",
        };
        let scale_layout = match self.scale_layout {
            PackedScaleLayout::PerTensor => "per-tensor".to_string(),
            PackedScaleLayout::PerRow => "per-row".to_string(),
            PackedScaleLayout::PerGroup { group_size } => format!("per-group:{group_size}"),
        };
        match self.equalizer_seed {
            Some(seed) => format!(
                "{format} {}x{} {}-bit words={} scales={} scale_layout={} equalizer_seed={seed}",
                self.rows,
                self.cols,
                self.bits_per_weight,
                self.packed_words.len(),
                self.scales.len(),
                scale_layout
            ),
            None => format!(
                "{format} {}x{} {}-bit words={} scales={} scale_layout={}",
                self.rows,
                self.cols,
                self.bits_per_weight,
                self.packed_words.len(),
                self.scales.len(),
                scale_layout
            ),
        }
    }
}

/// A quantized tensor: stores packed runtime weights alongside quantization
/// parameters. Conceptually represents an ndarray of signed low-bit values with
/// associated scale factors.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    /// Packed data (flat, row-major).
    packed: PackedStorage,
    /// Shape of the logical tensor.
    shape: Vec<usize>,
    /// Quantization parameters (scales, zero-points).
    quant_params: QuantParams,
}

impl PackedTensor {
    fn from_storage(
        packed: PackedStorage,
        shape: Vec<usize>,
        quant_params: QuantParams,
    ) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        let packed_len = match &packed {
            PackedStorage::Ternary(data) => data.len(),
            PackedStorage::Binary { data, .. } => data.len(),
        };
        if packed_len != total_elements {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![total_elements],
                got: vec![packed_len],
            });
        }
        if quant_params.original_shape != shape {
            return Err(OneBitError::ShapeMismatch {
                expected: shape,
                got: quant_params.original_shape,
            });
        }
        Ok(Self {
            packed,
            shape: quant_params.original_shape.clone(),
            quant_params,
        })
    }

    /// Rebuild a ternary packed tensor from serialized pieces.
    pub fn from_parts(
        packed: PackedTernary,
        shape: Vec<usize>,
        quant_params: QuantParams,
    ) -> Result<Self> {
        Self::from_storage(PackedStorage::Ternary(packed), shape, quant_params)
    }

    /// Rebuild a binary packed tensor from serialized pieces.
    pub fn from_binary_parts(
        packed: PackedBinary,
        shape: Vec<usize>,
        quant_params: QuantParams,
        equalizer_seed: u64,
    ) -> Result<Self> {
        Self::from_storage(
            PackedStorage::Binary {
                data: packed,
                equalizer_seed,
            },
            shape,
            quant_params,
        )
    }

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

        Self::from_storage(PackedStorage::Ternary(packed), shape, quant_params)
            .expect("ternary quantization should produce a valid packed tensor")
    }

    /// Create from an f32 ndarray by packing true 1-bit toggle weights.
    pub fn from_binary_ndarray(
        arr: &Array<f32, IxDyn>,
        granularity: QuantGranularity,
        equalizer_seed: u64,
    ) -> Self {
        let shape = arr.shape().to_vec();
        let flat: Vec<f32> = arr.iter().copied().collect();
        let quant_params = QuantParams::compute(
            &flat,
            &shape,
            &QuantConfig {
                granularity,
                use_zero_point: false,
                learnable_scale: false,
            },
        );
        let bits: Vec<bool> = flat
            .iter()
            .enumerate()
            .map(|(index, &value)| toggle_bit_for_sign(value >= 0.0, equalizer_seed, index))
            .collect();
        let packed = PackedBinary::from_bool_slice(&bits);

        Self::from_binary_parts(packed, shape, quant_params, equalizer_seed)
            .expect("binary quantization should produce a valid packed tensor")
    }

    /// Create from a 2D f32 array (most common: weight matrices).
    pub fn from_array2(arr: &Array2<f32>, config: &QuantConfig) -> Self {
        Self::from_ndarray(&arr.clone().into_dyn(), config)
    }

    /// Create from an ndarray that already contains exact unit ternary values.
    ///
    /// This keeps the logical values at {-1, 0, +1} by forcing unit scales
    /// instead of recomputing absmean scales from the data.
    pub fn from_unit_ternary_ndarray(
        arr: &Array<f32, IxDyn>,
        granularity: QuantGranularity,
    ) -> Result<Self> {
        let shape = arr.shape().to_vec();
        let flat: Vec<f32> = arr.iter().copied().collect();
        let ternary: Vec<TernaryWeight> = flat
            .iter()
            .map(|&value| match value {
                v if (v - 1.0).abs() < 1e-6 => Ok(TernaryWeight::Pos),
                v if (v + 1.0).abs() < 1e-6 => Ok(TernaryWeight::Neg),
                v if v.abs() < 1e-6 => Ok(TernaryWeight::Zero),
                _ => Err(OneBitError::Quantization(format!(
                    "expected exact unit ternary values, found {value}"
                ))),
            })
            .collect::<Result<Vec<_>>>()?;

        let scale_count = match granularity {
            QuantGranularity::PerTensor => 1,
            QuantGranularity::PerChannel => shape.first().copied().ok_or_else(|| {
                OneBitError::Quantization(
                    "per-channel ternary packing requires a non-empty shape".into(),
                )
            })?,
            QuantGranularity::PerGroup(group_size) => {
                if group_size == 0 {
                    return Err(OneBitError::Quantization(
                        "per-group ternary packing requires group_size > 0".into(),
                    ));
                }
                flat.len().div_ceil(group_size)
            }
        };

        Self::from_storage(
            PackedStorage::Ternary(PackedTernary::from_ternary_slice(&ternary)),
            shape.clone(),
            QuantParams {
                scales: vec![1.0; scale_count],
                zero_points: Vec::new(),
                original_shape: shape,
                granularity,
            },
        )
    }

    /// Dequantize back to f32 ndarray.
    ///
    /// Each ternary weight is multiplied by its group's scale factor.
    pub fn to_ndarray(&self) -> Array<f32, IxDyn> {
        let total = shape::num_elements(&self.shape);
        let mut flat = Vec::with_capacity(total);

        for i in 0..total {
            let scale = self.quant_params.scale_for_index(i);
            let value = match &self.packed {
                PackedStorage::Ternary(packed) => packed.get(i).to_f32() * scale,
                PackedStorage::Binary {
                    data,
                    equalizer_seed,
                } => effective_sign_from_toggle(data.get(i), *equalizer_seed, i) * scale,
            };
            flat.push(value);
        }

        Array::from_shape_vec(IxDyn(&self.shape), flat).expect("shape mismatch in dequantization")
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
        let sliced = arr.slice_axis(ndarray::Axis(dim), ndarray::Slice::from(start..end));
        let owned = sliced.to_owned();

        Ok(match &self.packed {
            PackedStorage::Ternary(_) => {
                let config = QuantConfig {
                    granularity: self.quant_params.granularity,
                    ..QuantConfig::default()
                };
                Self::from_ndarray(&owned, &config)
            }
            PackedStorage::Binary { equalizer_seed, .. } => {
                Self::from_binary_ndarray(&owned, self.quant_params.granularity, *equalizer_seed)
            }
        })
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

        Ok(match &self.packed {
            PackedStorage::Ternary(_) => {
                let config = QuantConfig {
                    granularity: self.quant_params.granularity,
                    ..QuantConfig::default()
                };
                Self::from_array2(&transposed, &config)
            }
            PackedStorage::Binary { equalizer_seed, .. } => Self::from_binary_ndarray(
                &transposed.into_dyn(),
                self.quant_params.granularity,
                *equalizer_seed,
            ),
        })
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
        let output: Vec<f32> = (0..m).map(|row| self.row_dot(row, &input_flat)).collect();

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
            .map(|row| {
                rhs_columns
                    .iter()
                    .map(|col| self.row_dot(row, col))
                    .collect()
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        let rows: Vec<Vec<f32>> = (0..m)
            .map(|row| {
                rhs_columns
                    .iter()
                    .map(|col| self.row_dot(row, col))
                    .collect()
            })
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

    /// Runtime weight format stored in this tensor.
    pub fn weight_format(&self) -> PackedWeightFormat {
        match &self.packed {
            PackedStorage::Ternary(_) => PackedWeightFormat::Ternary,
            PackedStorage::Binary { .. } => PackedWeightFormat::Binary,
        }
    }

    /// Access packed ternary data when the tensor stores ternary weights.
    pub fn ternary_data(&self) -> Option<&PackedTernary> {
        match &self.packed {
            PackedStorage::Ternary(data) => Some(data),
            PackedStorage::Binary { .. } => None,
        }
    }

    /// Access packed binary data when the tensor stores true 1-bit weights.
    pub fn binary_data(&self) -> Option<&PackedBinary> {
        match &self.packed {
            PackedStorage::Ternary(_) => None,
            PackedStorage::Binary { data, .. } => Some(data),
        }
    }

    /// Equalizer seed used for binary metadata-driven sign reconstruction.
    pub fn equalizer_seed(&self) -> Option<u64> {
        match &self.packed {
            PackedStorage::Ternary(_) => None,
            PackedStorage::Binary { equalizer_seed, .. } => Some(*equalizer_seed),
        }
    }

    /// Access quantization params.
    pub fn quant_params(&self) -> &QuantParams {
        &self.quant_params
    }

    /// Borrow this tensor as a 2D kernel-friendly packed layout.
    pub fn kernel_layout_2d(&self) -> Result<PackedTensorKernelLayout<'_>> {
        if self.shape.len() != 2 {
            return Err(OneBitError::TensorOp(format!(
                "kernel_layout_2d requires a 2D packed tensor, got {}D",
                self.shape.len()
            )));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let scale_layout = match self.quant_params.granularity {
            QuantGranularity::PerTensor => PackedScaleLayout::PerTensor,
            QuantGranularity::PerChannel => PackedScaleLayout::PerRow,
            QuantGranularity::PerGroup(group_size) => PackedScaleLayout::PerGroup { group_size },
        };

        let layout = match &self.packed {
            PackedStorage::Ternary(data) => PackedTensorKernelLayout {
                weight_format: PackedWeightFormat::Ternary,
                scale_layout,
                rows,
                cols,
                bits_per_weight: 2,
                packed_words: data.raw_data(),
                scales: &self.quant_params.scales,
                equalizer_seed: None,
            },
            PackedStorage::Binary {
                data,
                equalizer_seed,
            } => PackedTensorKernelLayout {
                weight_format: PackedWeightFormat::Binary,
                scale_layout,
                rows,
                cols,
                bits_per_weight: 1,
                packed_words: data.raw_data(),
                scales: &self.quant_params.scales,
                equalizer_seed: Some(*equalizer_seed),
            },
        };

        Ok(layout)
    }

    /// Memory usage in bytes (packed data only).
    pub fn memory_bytes(&self) -> usize {
        match &self.packed {
            PackedStorage::Ternary(data) => data.memory_bytes(),
            PackedStorage::Binary { data, .. } => data.memory_bytes(),
        }
    }

    fn row_dot(&self, row: usize, input: &[f32]) -> f32 {
        let n = self.shape[1];
        let row_start = row * n;

        match &self.packed {
            PackedStorage::Ternary(packed) => match self.quant_params.granularity {
                QuantGranularity::PerTensor => packed
                    .dot_slice_f32(row_start, input, self.quant_params.scales[0])
                    .expect("validated packed row slice should fit input"),
                QuantGranularity::PerChannel if self.shape.len() == 2 => packed
                    .dot_slice_f32(row_start, input, self.quant_params.scales[row])
                    .expect("validated packed row slice should fit input"),
                QuantGranularity::PerGroup(_) | QuantGranularity::PerChannel => {
                    let mut acc = 0.0f32;
                    for (col, &value) in input.iter().enumerate() {
                        let flat_idx = row_start + col;
                        let w = packed.get(flat_idx);
                        let scale = self.quant_params.scale_for_index(flat_idx);
                        acc += w.to_f32() * scale * value;
                    }
                    acc
                }
            },
            PackedStorage::Binary {
                data,
                equalizer_seed,
            } => {
                let mut acc = 0.0f32;
                for (col, &value) in input.iter().enumerate() {
                    let flat_idx = row_start + col;
                    let sign =
                        effective_sign_from_toggle(data.get(flat_idx), *equalizer_seed, flat_idx);
                    let scale = self.quant_params.scale_for_index(flat_idx);
                    acc += sign * scale * value;
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
    fn test_from_unit_ternary_ndarray_keeps_unit_scale() {
        let arr = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let packed =
            PackedTensor::from_unit_ternary_ndarray(&arr, QuantGranularity::PerTensor).unwrap();
        assert_eq!(packed.quant_params().scales, vec![1.0]);
        assert_eq!(packed.to_ndarray(), arr);
    }

    #[test]
    fn test_from_parts_roundtrip() {
        let arr = array![[1.0f32, -1.0], [0.0, 1.0]].into_dyn();
        let packed =
            PackedTensor::from_unit_ternary_ndarray(&arr, QuantGranularity::PerTensor).unwrap();
        let rebuilt = PackedTensor::from_parts(
            packed.ternary_data().unwrap().clone(),
            packed.shape().to_vec(),
            packed.quant_params().clone(),
        )
        .unwrap();
        assert_eq!(rebuilt.to_ndarray(), arr);
    }

    #[test]
    fn test_binary_roundtrip_preserves_effective_signs() {
        let arr = array![[2.0f32, -3.0], [-0.5, 1.5]].into_dyn();
        let seed = 0x1B17_EA11_u64;
        let packed = PackedTensor::from_binary_ndarray(&arr, QuantGranularity::PerTensor, seed);

        assert_eq!(packed.weight_format(), PackedWeightFormat::Binary);
        assert_eq!(packed.equalizer_seed(), Some(seed));
        let rebuilt = packed.to_ndarray();

        for (original, restored) in arr.iter().zip(rebuilt.iter()) {
            assert_eq!(original.is_sign_positive(), restored.is_sign_positive());
        }
    }

    #[test]
    fn test_kernel_layout_2d_binary_exposes_equalizer_metadata() {
        let arr = array![[2.0f32, -3.0], [-0.5, 1.5]].into_dyn();
        let seed = 0x1B17_EA11_u64;
        let packed = PackedTensor::from_binary_ndarray(&arr, QuantGranularity::PerTensor, seed);

        let layout = packed.kernel_layout_2d().unwrap();
        assert_eq!(layout.weight_format, PackedWeightFormat::Binary);
        assert_eq!(layout.scale_layout, PackedScaleLayout::PerTensor);
        assert_eq!(layout.rows, 2);
        assert_eq!(layout.cols, 2);
        assert_eq!(layout.bits_per_weight, 1);
        assert_eq!(layout.equalizer_seed, Some(seed));
        assert_eq!(layout.scales.len(), 1);
        assert!(!layout.packed_words.is_empty());
    }

    #[test]
    fn test_kernel_layout_2d_per_channel_maps_to_per_row() {
        let arr = array![[10.0f32, -10.0, 0.0], [0.1, -0.1, 0.0]].into_dyn();
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_channel());

        let layout = packed.kernel_layout_2d().unwrap();
        assert_eq!(layout.weight_format, PackedWeightFormat::Ternary);
        assert_eq!(layout.scale_layout, PackedScaleLayout::PerRow);
        assert_eq!(layout.rows, 2);
        assert_eq!(layout.cols, 3);
        assert_eq!(layout.bits_per_weight, 2);
        assert_eq!(layout.equalizer_seed, None);
        assert_eq!(layout.scales.len(), 2);
    }

    #[test]
    fn test_kernel_layout_2d_rejects_non_matrix() {
        let arr = Array::from_elem(IxDyn(&[2, 2, 2]), 1.0f32);
        let packed = PackedTensor::from_ndarray(&arr, &QuantConfig::per_tensor());
        assert!(packed.kernel_layout_2d().is_err());
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
        let arr = array![[10.0f32, -10.0, 0.0], [0.1, -0.1, 0.0]].into_dyn();
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
