//! SafeTensors format support.
//!
//! Provides save/load for model parameters using the SafeTensors format,
//! a simple, safe format for storing tensors. Requires the `safetensors-io` feature.

use std::io::Write;

use ndarray::{Array, IxDyn};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use safetensors::serialize;

use crate::Result;
use crate::error::OneBitError;
use crate::nn::Parameter;

/// Save parameters to SafeTensors format.
///
/// Each parameter is stored as an f32 tensor keyed by its name.
pub fn save_safetensors<W: Write>(params: &[&Parameter], mut w: W) -> Result<()> {
    let mut data_buffers: Vec<Vec<u8>> = Vec::with_capacity(params.len());

    // Pre-convert all data to bytes
    for param in params {
        let flat: Vec<f32> = param.data.iter().copied().collect();
        let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
        data_buffers.push(bytes);
    }

    // Create TensorViews referencing the buffers, collected into owned tuples
    let mut tensors: Vec<(String, TensorView<'_>)> = Vec::with_capacity(params.len());
    for (i, param) in params.iter().enumerate() {
        let shape: Vec<usize> = param.data.shape().to_vec();
        let view = TensorView::new(Dtype::F32, shape, &data_buffers[i])
            .map_err(|e| OneBitError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to create tensor view for '{}': {e}", param.name),
            )))?;
        tensors.push((param.name.clone(), view));
    }

    let serialized = serialize(tensors, &None)
        .map_err(|e| OneBitError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to serialize safetensors: {e}"),
        )))?;

    w.write_all(&serialized)?;
    Ok(())
}

/// Load parameters from SafeTensors format.
///
/// Returns named tensors as (name, shape, f32 data) tuples.
pub fn load_safetensors(data: &[u8]) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let tensors = SafeTensors::deserialize(data)
        .map_err(|e| OneBitError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to deserialize safetensors: {e}"),
        )))?;

    let mut result = Vec::new();

    for (name, tensor) in tensors.tensors() {
        if tensor.dtype() != Dtype::F32 {
            return Err(OneBitError::Config(format!(
                "Unsupported dtype {:?} for tensor '{}' (expected F32)",
                tensor.dtype(), name
            )));
        }

        let shape = tensor.shape().to_vec();
        let bytes = tensor.data();
        if bytes.len() % 4 != 0 {
            return Err(OneBitError::Config(format!(
                "Tensor '{}' data length {} not a multiple of 4",
                name, bytes.len()
            )));
        }

        let f32_data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        result.push((name.to_string(), shape, f32_data));
    }

    Ok(result)
}

/// Restore SafeTensors data into existing parameters (matched by name).
pub fn restore_safetensors_into(
    data: &[u8],
    params: &mut [&mut Parameter],
) -> Result<()> {
    let loaded = load_safetensors(data)?;

    for (name, shape, f32_data) in &loaded {
        if let Some(param) = params.iter_mut().find(|p| &p.name == name) {
            let arr = Array::from_shape_vec(IxDyn(shape), f32_data.clone())
                .map_err(|e| OneBitError::Config(format!(
                    "Failed to reshape tensor '{}': {e}", name
                )))?;
            param.data = arr;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_roundtrip() {
        let p1 = Parameter::new(
            "weight",
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let p2 = Parameter::new(
            "bias",
            Array::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap(),
        );

        let mut buf = Vec::new();
        save_safetensors(&[&p1, &p2], &mut buf).unwrap();

        let loaded = load_safetensors(&buf).unwrap();
        assert_eq!(loaded.len(), 2);

        // Find by name (order not guaranteed)
        let w = loaded.iter().find(|(n, _, _)| n == "weight").unwrap();
        assert_eq!(w.1, vec![2, 3]);
        assert_eq!(w.2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let b = loaded.iter().find(|(n, _, _)| n == "bias").unwrap();
        assert_eq!(b.1, vec![3]);
        assert_eq!(b.2, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_safetensors_restore_into() {
        let p1 = Parameter::new(
            "w",
            Array::from_shape_vec(IxDyn(&[2]), vec![10.0, 20.0]).unwrap(),
        );

        let mut buf = Vec::new();
        save_safetensors(&[&p1], &mut buf).unwrap();

        let mut p_new = Parameter::new(
            "w",
            Array::from_elem(IxDyn(&[2]), 0.0f32),
        );

        restore_safetensors_into(&buf, &mut [&mut p_new]).unwrap();
        assert_eq!(p_new.data.as_slice().unwrap(), &[10.0, 20.0]);
    }
}
