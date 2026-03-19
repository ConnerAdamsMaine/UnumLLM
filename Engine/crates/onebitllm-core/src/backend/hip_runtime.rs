use std::path::PathBuf;

use libloading::{Library, Symbol};
use ndarray::{Array, Array2, Ix2, IxDyn};

use crate::error::OneBitError;
use crate::tensor::{
    PackedScaleLayout, PackedTensor, PackedTensorKernelLayout, PackedWeightFormat,
};
use crate::Result;

const HIP_LAYOUT_ABI_VERSION: u32 = 1;
const HIP_WEIGHT_FORMAT_TERNARY: u32 = 0;
const HIP_WEIGHT_FORMAT_BINARY: u32 = 1;
const HIP_SCALE_LAYOUT_PER_TENSOR: u32 = 0;
const HIP_SCALE_LAYOUT_PER_ROW: u32 = 1;
const HIP_SCALE_LAYOUT_PER_GROUP: u32 = 2;

type PackedDenseLeftFn = unsafe extern "C" fn(
    packed: *const HipPackedMatrixLayout,
    lhs: *const f32,
    lhs_rows: u64,
    lhs_cols: u64,
    out: *mut f32,
) -> i32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HipPackedMatrixLayout {
    abi_version: u32,
    weight_format: u32,
    scale_layout: u32,
    bits_per_weight: u32,
    rows: u64,
    cols: u64,
    packed_word_count: u64,
    packed_words: *const u64,
    scale_count: u64,
    scales: *const f32,
    group_size: u64,
    equalizer_seed: u64,
    has_equalizer_seed: u32,
}

fn ffi_layout(layout: &PackedTensorKernelLayout<'_>) -> HipPackedMatrixLayout {
    let (scale_layout, group_size) = match layout.scale_layout {
        PackedScaleLayout::PerTensor => (HIP_SCALE_LAYOUT_PER_TENSOR, 0),
        PackedScaleLayout::PerRow => (HIP_SCALE_LAYOUT_PER_ROW, 0),
        PackedScaleLayout::PerGroup { group_size } => (HIP_SCALE_LAYOUT_PER_GROUP, group_size),
    };
    let (weight_format, equalizer_seed, has_equalizer_seed) = match layout.weight_format {
        PackedWeightFormat::Ternary => (HIP_WEIGHT_FORMAT_TERNARY, 0, 0),
        PackedWeightFormat::Binary => (
            HIP_WEIGHT_FORMAT_BINARY,
            layout.equalizer_seed.unwrap_or(0),
            u32::from(layout.equalizer_seed.is_some()),
        ),
    };

    HipPackedMatrixLayout {
        abi_version: HIP_LAYOUT_ABI_VERSION,
        weight_format,
        scale_layout,
        bits_per_weight: layout.bits_per_weight as u32,
        rows: layout.rows as u64,
        cols: layout.cols as u64,
        packed_word_count: layout.packed_words.len() as u64,
        packed_words: layout.packed_words.as_ptr(),
        scale_count: layout.scales.len() as u64,
        scales: layout.scales.as_ptr(),
        group_size: group_size as u64,
        equalizer_seed,
        has_equalizer_seed,
    }
}

fn kernel_library_path() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("ONEBITLLM_HIP_KERNEL_LIB") {
        return Ok(PathBuf::from(path));
    }

    if let Some(path) = option_env!("ONEBITLLM_HIP_KERNEL_LIB") {
        return Ok(PathBuf::from(path));
    }

    Err(OneBitError::Other(
        "ROCm backend requested, but no HIP kernel library was configured. Build with `--features rocm` on a machine with `hipcc`, or set `ONEBITLLM_HIP_KERNEL_LIB` to a compiled kernel library.".into(),
    ))
}

pub fn packed_matmul_dense_left_transposed(
    packed: &PackedTensor,
    lhs: &Array<f32, IxDyn>,
) -> Result<Array<f32, IxDyn>> {
    let layout = packed.kernel_layout_2d()?;
    let lhs2 = lhs
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|e| OneBitError::TensorOp(e.to_string()))?;

    if lhs2.shape()[1] != layout.cols {
        return Err(OneBitError::ShapeMismatch {
            expected: vec![layout.cols],
            got: vec![lhs2.shape()[1]],
        });
    }

    let lhs_owned = lhs2.to_owned();
    let batch = lhs_owned.shape()[0];
    let out_features = layout.rows;
    let mut output = vec![0.0f32; batch * out_features];
    let ffi = ffi_layout(&layout);
    let path = kernel_library_path()?;
    if !path.exists() {
        return Err(OneBitError::Other(format!(
            "HIP kernel library path does not exist: {}",
            path.display()
        )));
    }

    unsafe {
        let library = Library::new(&path).map_err(|err| {
            OneBitError::Other(format!(
                "failed to load HIP kernel library {}: {err}",
                path.display()
            ))
        })?;
        let entrypoint: Symbol<PackedDenseLeftFn> = library
            .get(b"onebitllm_hip_packed_matmul_dense_left_transposed_v1")
            .map_err(|err| {
                OneBitError::Other(format!(
                    "HIP kernel library {} is missing `onebitllm_hip_packed_matmul_dense_left_transposed_v1`: {err}",
                    path.display()
                ))
            })?;

        let status = entrypoint(
            &ffi,
            lhs_owned.as_ptr(),
            batch as u64,
            layout.cols as u64,
            output.as_mut_ptr(),
        );
        if status != 0 {
            return Err(OneBitError::Other(format!(
                "HIP packed dense-left matmul entrypoint returned error code {status} for {}",
                layout.describe()
            )));
        }
    }

    Ok(Array2::from_shape_vec((batch, out_features), output)
        .expect("HIP packed matmul output shape should be valid")
        .into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{QuantConfig, QuantGranularity};
    use ndarray::array;

    #[test]
    fn test_ffi_layout_binary_preserves_equalizer_seed() {
        let packed = PackedTensor::from_binary_ndarray(
            &array![[1.0f32, -1.0]].into_dyn(),
            QuantGranularity::PerTensor,
            99,
        );
        let layout = packed.kernel_layout_2d().unwrap();
        let ffi = ffi_layout(&layout);
        assert_eq!(ffi.abi_version, HIP_LAYOUT_ABI_VERSION);
        assert_eq!(ffi.weight_format, HIP_WEIGHT_FORMAT_BINARY);
        assert_eq!(ffi.scale_layout, HIP_SCALE_LAYOUT_PER_TENSOR);
        assert_eq!(ffi.equalizer_seed, 99);
        assert_eq!(ffi.has_equalizer_seed, 1);
    }

    #[test]
    fn test_ffi_layout_per_row_uses_row_scale_layout() {
        let packed = PackedTensor::from_ndarray(
            &array![[10.0f32, -10.0], [1.0, -1.0]].into_dyn(),
            &QuantConfig::per_channel(),
        );
        let layout = packed.kernel_layout_2d().unwrap();
        let ffi = ffi_layout(&layout);
        assert_eq!(ffi.weight_format, HIP_WEIGHT_FORMAT_TERNARY);
        assert_eq!(ffi.scale_layout, HIP_SCALE_LAYOUT_PER_ROW);
        assert_eq!(ffi.has_equalizer_seed, 0);
    }
}
