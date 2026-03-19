//! Checkpoint save/load utilities.
//!
//! Provides basic serialization of model parameters for saving and
//! restoring training state. Uses a simple binary format.

use crate::nn::Parameter;
use ndarray::{Array, IxDyn};
use std::io::{Read, Write};

/// A serializable checkpoint containing parameter names, shapes, and data.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Named parameter entries.
    pub entries: Vec<CheckpointEntry>,
    /// Training step at which this checkpoint was saved.
    pub step: usize,
}

/// A single parameter entry in a checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    /// Parameter name.
    pub name: String,
    /// Shape of the parameter tensor.
    pub shape: Vec<usize>,
    /// Flat f32 data.
    pub data: Vec<f32>,
}

impl Checkpoint {
    /// Create a checkpoint from a list of parameters.
    pub fn from_parameters(params: &[&Parameter], step: usize) -> Self {
        let entries = params
            .iter()
            .map(|p| CheckpointEntry {
                name: p.name.clone(),
                shape: p.data.shape().to_vec(),
                data: p.data.iter().copied().collect(),
            })
            .collect();

        Self { entries, step }
    }

    /// Restore parameters from this checkpoint.
    ///
    /// Parameters are matched by name. Returns an error if a parameter
    /// name in the checkpoint doesn't match any provided parameter.
    pub fn restore_into(&self, params: &mut [&mut Parameter]) -> crate::Result<()> {
        for entry in &self.entries {
            let param = params
                .iter_mut()
                .find(|p| p.name == entry.name)
                .ok_or_else(|| {
                    crate::error::OneBitError::Training(format!(
                        "Checkpoint parameter '{}' not found in model",
                        entry.name
                    ))
                })?;

            let data =
                Array::from_shape_vec(IxDyn(&entry.shape), entry.data.clone()).map_err(|e| {
                    crate::error::OneBitError::Training(format!(
                        "Failed to restore parameter '{}': {e}",
                        entry.name
                    ))
                })?;

            param.data = data;
        }
        Ok(())
    }

    /// Serialize the checkpoint to a writer in a simple binary format.
    ///
    /// Format:
    /// - magic: 4 bytes "OB1C"
    /// - step: 8 bytes (u64 little-endian)
    /// - num_entries: 4 bytes (u32 LE)
    /// - for each entry:
    ///   - name_len: 4 bytes (u32 LE)
    ///   - name: name_len bytes (UTF-8)
    ///   - ndim: 4 bytes (u32 LE)
    ///   - shape: ndim * 8 bytes (u64 LE each)
    ///   - data: product(shape) * 4 bytes (f32 LE each)
    pub fn save<W: Write>(&self, mut w: W) -> crate::Result<()> {
        // Magic
        w.write_all(b"OB1C")?;
        // Step
        w.write_all(&(self.step as u64).to_le_bytes())?;
        // Num entries
        w.write_all(&(self.entries.len() as u32).to_le_bytes())?;

        for entry in &self.entries {
            // Name
            let name_bytes = entry.name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            w.write_all(name_bytes)?;

            // Shape
            w.write_all(&(entry.shape.len() as u32).to_le_bytes())?;
            for &dim in &entry.shape {
                w.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Data
            for &val in &entry.data {
                w.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Deserialize a checkpoint from a reader.
    pub fn load<R: Read>(mut r: R) -> crate::Result<Self> {
        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"OB1C" {
            return Err(crate::error::OneBitError::Training(
                "Invalid checkpoint magic bytes".into(),
            ));
        }

        // Step
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let step = u64::from_le_bytes(buf8) as usize;

        // Num entries
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let num_entries = u32::from_le_bytes(buf4) as usize;

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            // Name
            r.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf).map_err(|e| {
                crate::error::OneBitError::Training(format!("Invalid UTF-8 in parameter name: {e}"))
            })?;

            // Shape
            r.read_exact(&mut buf4)?;
            let ndim = u32::from_le_bytes(buf4) as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                r.read_exact(&mut buf8)?;
                shape.push(u64::from_le_bytes(buf8) as usize);
            }

            // Data
            let num_elements: usize = shape.iter().product();
            let mut data = Vec::with_capacity(num_elements);
            let mut fbuf = [0u8; 4];
            for _ in 0..num_elements {
                r.read_exact(&mut fbuf)?;
                data.push(f32::from_le_bytes(fbuf));
            }

            entries.push(CheckpointEntry { name, shape, data });
        }

        Ok(Self { entries, step })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_checkpoint_roundtrip() {
        let p1 = Parameter::new(
            "layer1.weight",
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        );
        let p2 = Parameter::new(
            "layer1.bias",
            Array::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
        );

        let ckpt = Checkpoint::from_parameters(&[&p1, &p2], 42);
        assert_eq!(ckpt.step, 42);
        assert_eq!(ckpt.entries.len(), 2);

        // Serialize
        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        // Deserialize
        let loaded = Checkpoint::load(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.step, 42);
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.entries[0].name, "layer1.weight");
        assert_eq!(loaded.entries[0].shape, vec![2, 3]);
        assert_eq!(loaded.entries[0].data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(loaded.entries[1].name, "layer1.bias");
        assert_eq!(loaded.entries[1].data, vec![0.1, 0.2]);
    }

    #[test]
    fn test_checkpoint_restore_into() {
        let p1 = Parameter::new(
            "w",
            Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        );
        let ckpt = Checkpoint::from_parameters(&[&p1], 10);

        // Create new parameter with different data
        let mut p_new = Parameter::new("w", Array::from_elem(IxDyn(&[3]), 0.0f32));

        ckpt.restore_into(&mut [&mut p_new]).unwrap();

        assert_eq!(p_new.data.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_restore_missing_param() {
        let p1 = Parameter::new("nonexistent", Array::from_elem(IxDyn(&[1]), 1.0f32));
        let ckpt = Checkpoint::from_parameters(&[&p1], 0);

        let mut p_new = Parameter::new("different_name", Array::from_elem(IxDyn(&[1]), 0.0f32));

        assert!(ckpt.restore_into(&mut [&mut p_new]).is_err());
    }

    #[test]
    fn test_checkpoint_invalid_magic() {
        let bad_data = b"BADx";
        let result = Checkpoint::load(Cursor::new(bad_data));
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_empty_params() {
        let ckpt = Checkpoint::from_parameters(&[], 0);
        assert!(ckpt.entries.is_empty());

        let mut buf = Vec::new();
        ckpt.save(&mut buf).unwrap();

        let loaded = Checkpoint::load(Cursor::new(&buf)).unwrap();
        assert!(loaded.entries.is_empty());
        assert_eq!(loaded.step, 0);
    }
}
