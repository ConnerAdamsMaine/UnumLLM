//! Custom `.obm` (OneBit Model) binary format.
//!
//! A compact binary format designed for 1-bit quantized models. Stores both
//! model configuration and weight tensors (f32 or bitpacked ternary) in a
//! single file with a clear header for fast loading.
//!
//! ## Format Layout
//!
//! ```text
//! [Magic: 4 bytes "OBM1"]
//! [Version: u32 LE]
//! [Config JSON length: u64 LE]
//! [Config JSON: N bytes]
//! [Num tensors: u32 LE]
//! For each tensor:
//!   [Name length: u32 LE]
//!   [Name: N bytes UTF-8]
//!   [Format: u8] (0=f32, 1=bitpacked_ternary)
//!   [Ndim: u32 LE]
//!   [Shape: ndim × u64 LE]
//!   [Data length: u64 LE]  (in bytes)
//!   [Data: N bytes]
//! ```

use std::io::{Read, Write};

use crate::Result;
use crate::error::OneBitError;
use super::config::ModelConfig;

const OBM_MAGIC: &[u8; 4] = b"OBM1";
const OBM_VERSION: u32 = 1;

/// Tensor data format tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorFormat {
    /// Standard f32 data.
    Float32 = 0,
    /// Bitpacked ternary (2 bits per weight, packed into u64s).
    BitpackedTernary = 1,
}

impl TensorFormat {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(TensorFormat::Float32),
            1 => Ok(TensorFormat::BitpackedTernary),
            _ => Err(OneBitError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown tensor format: {v}"),
            ))),
        }
    }
}

/// Header metadata for an OBM file.
#[derive(Debug, Clone)]
pub struct ObmHeader {
    /// Format version.
    pub version: u32,
    /// Model configuration.
    pub config: ModelConfig,
    /// Number of tensors in the file.
    pub num_tensors: u32,
}

/// A single tensor entry in an OBM file.
#[derive(Debug, Clone)]
pub struct ObmTensor {
    /// Tensor name (e.g., "layers.0.attention.q_proj.weight").
    pub name: String,
    /// Storage format.
    pub format: TensorFormat,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Raw data bytes.
    pub data: Vec<u8>,
}

impl ObmTensor {
    /// Create an f32 tensor entry from name, shape, and flat f32 data.
    pub fn from_f32(name: impl Into<String>, shape: Vec<usize>, data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.into(),
            format: TensorFormat::Float32,
            shape,
            data: bytes,
        }
    }

    /// Create a bitpacked ternary tensor entry from name, shape, and packed u64 data.
    pub fn from_packed(name: impl Into<String>, shape: Vec<usize>, packed: &[u64]) -> Self {
        let bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.into(),
            format: TensorFormat::BitpackedTernary,
            shape,
            data: bytes,
        }
    }

    /// Interpret data as f32 slice (only valid if format is Float32).
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.format != TensorFormat::Float32 {
            return Err(OneBitError::Config(
                "Tensor is not in f32 format".into(),
            ));
        }
        if self.data.len() % 4 != 0 {
            return Err(OneBitError::Config(
                "f32 tensor data length not a multiple of 4".into(),
            ));
        }
        Ok(self.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Interpret data as u64 slice (only valid if format is BitpackedTernary).
    pub fn as_packed_u64(&self) -> Result<Vec<u64>> {
        if self.format != TensorFormat::BitpackedTernary {
            return Err(OneBitError::Config(
                "Tensor is not in bitpacked format".into(),
            ));
        }
        if self.data.len() % 8 != 0 {
            return Err(OneBitError::Config(
                "Packed tensor data length not a multiple of 8".into(),
            ));
        }
        Ok(self.data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }
}

/// OBM file container — holds config and all tensors.
#[derive(Debug, Clone)]
pub struct ObmFile {
    /// File header with config.
    pub header: ObmHeader,
    /// All tensor entries.
    pub tensors: Vec<ObmTensor>,
}

impl ObmFile {
    /// Create a new OBM file from a config and tensor list.
    pub fn new(config: ModelConfig, tensors: Vec<ObmTensor>) -> Self {
        Self {
            header: ObmHeader {
                version: OBM_VERSION,
                config,
                num_tensors: tensors.len() as u32,
            },
            tensors,
        }
    }

    /// Serialize the entire OBM file to a writer.
    pub fn save<W: Write>(&self, mut w: W) -> Result<()> {
        // Magic
        w.write_all(OBM_MAGIC)?;

        // Version
        w.write_all(&self.header.version.to_le_bytes())?;

        // Config as JSON
        let mut config_buf = Vec::new();
        self.header.config.save_json(&mut config_buf)?;
        w.write_all(&(config_buf.len() as u64).to_le_bytes())?;
        w.write_all(&config_buf)?;

        // Num tensors
        w.write_all(&self.header.num_tensors.to_le_bytes())?;

        // Each tensor
        for tensor in &self.tensors {
            // Name
            let name_bytes = tensor.name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            w.write_all(name_bytes)?;

            // Format
            w.write_all(&[tensor.format as u8])?;

            // Shape
            w.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
            for &dim in &tensor.shape {
                w.write_all(&(dim as u64).to_le_bytes())?;
            }

            // Data
            w.write_all(&(tensor.data.len() as u64).to_le_bytes())?;
            w.write_all(&tensor.data)?;
        }

        Ok(())
    }

    /// Deserialize an OBM file from a reader.
    pub fn load<R: Read>(mut r: R) -> Result<Self> {
        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != OBM_MAGIC {
            return Err(OneBitError::Config(format!(
                "Invalid OBM magic: expected {:?}, got {:?}",
                OBM_MAGIC, magic
            )));
        }

        // Version
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != OBM_VERSION {
            return Err(OneBitError::Config(format!(
                "Unsupported OBM version: {version} (expected {OBM_VERSION})"
            )));
        }

        // Config JSON
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let config_len = u64::from_le_bytes(buf8) as usize;
        let mut config_buf = vec![0u8; config_len];
        r.read_exact(&mut config_buf)?;
        let config_str = String::from_utf8(config_buf).map_err(|e| {
            OneBitError::Config(format!("Invalid UTF-8 in OBM config: {e}"))
        })?;
        let config = ModelConfig::from_json_str(&config_str)?;

        // Num tensors
        r.read_exact(&mut buf4)?;
        let num_tensors = u32::from_le_bytes(buf4);

        // Tensors
        let mut tensors = Vec::with_capacity(num_tensors as usize);
        for _ in 0..num_tensors {
            // Name
            r.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf).map_err(|e| {
                OneBitError::Config(format!("Invalid UTF-8 in tensor name: {e}"))
            })?;

            // Format
            let mut fmt_byte = [0u8; 1];
            r.read_exact(&mut fmt_byte)?;
            let format = TensorFormat::from_u8(fmt_byte[0])?;

            // Shape
            r.read_exact(&mut buf4)?;
            let ndim = u32::from_le_bytes(buf4) as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                r.read_exact(&mut buf8)?;
                shape.push(u64::from_le_bytes(buf8) as usize);
            }

            // Data
            r.read_exact(&mut buf8)?;
            let data_len = u64::from_le_bytes(buf8) as usize;
            let mut data = vec![0u8; data_len];
            r.read_exact(&mut data)?;

            tensors.push(ObmTensor {
                name,
                format,
                shape,
                data,
            });
        }

        Ok(Self {
            header: ObmHeader {
                version,
                config,
                num_tensors,
            },
            tensors,
        })
    }

    /// Find a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&ObmTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_obm_roundtrip_f32() {
        let config = ModelConfig::default();
        let data = vec![1.0f32, 2.0, 3.0, -1.0, 0.0, 0.5];
        let tensor = ObmTensor::from_f32("layer.weight", vec![2, 3], &data);
        let obm = ObmFile::new(config.clone(), vec![tensor]);

        let mut buf = Vec::new();
        obm.save(&mut buf).unwrap();

        let loaded = ObmFile::load(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.header.version, OBM_VERSION);
        assert_eq!(loaded.header.config.architecture, config.architecture);
        assert_eq!(loaded.tensors.len(), 1);

        let t = &loaded.tensors[0];
        assert_eq!(t.name, "layer.weight");
        assert_eq!(t.format, TensorFormat::Float32);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.as_f32().unwrap(), data);
    }

    #[test]
    fn test_obm_roundtrip_packed() {
        let config = ModelConfig::default();
        let packed = vec![0xDEADBEEF_u64, 0xCAFEBABE];
        let tensor = ObmTensor::from_packed("layer.packed_weight", vec![64], &packed);
        let obm = ObmFile::new(config, vec![tensor]);

        let mut buf = Vec::new();
        obm.save(&mut buf).unwrap();

        let loaded = ObmFile::load(Cursor::new(&buf)).unwrap();
        let t = &loaded.tensors[0];
        assert_eq!(t.format, TensorFormat::BitpackedTernary);
        assert_eq!(t.as_packed_u64().unwrap(), packed);
    }

    #[test]
    fn test_obm_multiple_tensors() {
        let config = ModelConfig::default();
        let t1 = ObmTensor::from_f32("w1", vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let t2 = ObmTensor::from_f32("w2", vec![2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let t3 = ObmTensor::from_packed("w3", vec![32], &[0xFF]);
        let obm = ObmFile::new(config, vec![t1, t2, t3]);

        let mut buf = Vec::new();
        obm.save(&mut buf).unwrap();

        let loaded = ObmFile::load(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.tensors.len(), 3);
        assert_eq!(loaded.get_tensor("w1").unwrap().name, "w1");
        assert_eq!(loaded.get_tensor("w2").unwrap().shape, vec![2, 2]);
        assert_eq!(loaded.get_tensor("w3").unwrap().format, TensorFormat::BitpackedTernary);
        assert!(loaded.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_obm_empty_file() {
        let config = ModelConfig::default();
        let obm = ObmFile::new(config, vec![]);

        let mut buf = Vec::new();
        obm.save(&mut buf).unwrap();

        let loaded = ObmFile::load(Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.tensors.len(), 0);
    }

    #[test]
    fn test_obm_invalid_magic() {
        let bad = b"BAD!extra";
        assert!(ObmFile::load(Cursor::new(bad)).is_err());
    }

    #[test]
    fn test_tensor_format_conversion() {
        let t = ObmTensor::from_f32("x", vec![2], &[1.0, 2.0]);
        assert!(t.as_f32().is_ok());
        assert!(t.as_packed_u64().is_err());

        let t = ObmTensor::from_packed("y", vec![32], &[0u64]);
        assert!(t.as_packed_u64().is_ok());
        assert!(t.as_f32().is_err());
    }
}
