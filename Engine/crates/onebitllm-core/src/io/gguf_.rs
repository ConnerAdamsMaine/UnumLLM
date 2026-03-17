//! GGUF format support (read-only metadata parsing).
//!
//! GGUF is a binary format used by llama.cpp and related projects.
//! This module provides basic header/metadata parsing to read model
//! configuration from GGUF files. Full tensor loading is not yet supported
//! since GGUF uses different quantization schemes than our ternary format.

use std::io::Read;

use crate::Result;
use crate::error::OneBitError;

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Try to interpret as a string.
    pub fn as_str(&self) -> Option<&str> {
        if let GgufValue::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    /// Try to interpret as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Uint8(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to interpret as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Uint32(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to interpret as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }
}

/// Parsed GGUF file header with metadata.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// GGUF version (typically 2 or 3).
    pub version: u32,
    /// Number of tensors described in the file.
    pub tensor_count: u64,
    /// Metadata key-value pairs.
    pub metadata: Vec<(String, GgufValue)>,
}

impl GgufHeader {
    /// Parse a GGUF header from a reader.
    ///
    /// Only reads the header and metadata section, not tensor data.
    pub fn read<R: Read>(mut r: R) -> Result<Self> {
        // Magic
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != GGUF_MAGIC {
            return Err(OneBitError::Config(format!(
                "Invalid GGUF magic: 0x{magic:08X} (expected 0x{GGUF_MAGIC:08X})"
            )));
        }

        // Version
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);

        // Tensor count
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        let tensor_count = u64::from_le_bytes(buf8);

        // Metadata KV count
        r.read_exact(&mut buf8)?;
        let metadata_kv_count = u64::from_le_bytes(buf8);

        // Read metadata entries
        let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = read_gguf_string(&mut r)?;
            let value = read_gguf_value(&mut r)?;
            metadata.push((key, value));
        }

        Ok(Self {
            version,
            tensor_count,
            metadata,
        })
    }

    /// Look up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Get a metadata string by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    /// Get a metadata u32 by key.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)?.as_u32()
    }
}

/// Read a GGUF-format string (u64 length + bytes).
fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let mut buf8 = [0u8; 8];
    r.read_exact(&mut buf8)?;
    let len = u64::from_le_bytes(buf8) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| {
        OneBitError::Config(format!("Invalid UTF-8 in GGUF string: {e}"))
    })
}

/// Read a GGUF typed value.
fn read_gguf_value<R: Read>(r: &mut R) -> Result<GgufValue> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let value_type = u32::from_le_bytes(buf4);

    match value_type {
        0 => { // UINT8
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Uint8(buf[0]))
        }
        1 => { // INT8
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Int8(buf[0] as i8))
        }
        2 => { // UINT16
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
        }
        3 => { // INT16
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
        }
        4 => { // UINT32
            r.read_exact(&mut buf4)?;
            Ok(GgufValue::Uint32(u32::from_le_bytes(buf4)))
        }
        5 => { // INT32
            r.read_exact(&mut buf4)?;
            Ok(GgufValue::Int32(i32::from_le_bytes(buf4)))
        }
        6 => { // FLOAT32
            r.read_exact(&mut buf4)?;
            Ok(GgufValue::Float32(f32::from_le_bytes(buf4)))
        }
        7 => { // BOOL
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(GgufValue::Bool(buf[0] != 0))
        }
        8 => { // STRING
            Ok(GgufValue::String(read_gguf_string(r)?))
        }
        9 => { // ARRAY
            // Read element type and count
            r.read_exact(&mut buf4)?;
            let _elem_type = u32::from_le_bytes(buf4);
            let mut buf8 = [0u8; 8];
            r.read_exact(&mut buf8)?;
            let count = u64::from_le_bytes(buf8) as usize;

            // For simplicity, read each element as a full typed value
            // (In practice, arrays are homogeneous, but we handle them generically)
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                // Re-read with the element type — construct a mini buffer
                let val = read_gguf_value_of_type(r, _elem_type)?;
                values.push(val);
            }
            Ok(GgufValue::Array(values))
        }
        10 => { // UINT64
            let mut buf8 = [0u8; 8];
            r.read_exact(&mut buf8)?;
            Ok(GgufValue::Uint64(u64::from_le_bytes(buf8)))
        }
        11 => { // INT64
            let mut buf8 = [0u8; 8];
            r.read_exact(&mut buf8)?;
            Ok(GgufValue::Int64(i64::from_le_bytes(buf8)))
        }
        12 => { // FLOAT64
            let mut buf8 = [0u8; 8];
            r.read_exact(&mut buf8)?;
            Ok(GgufValue::Float64(f64::from_le_bytes(buf8)))
        }
        _ => Err(OneBitError::Config(format!(
            "Unknown GGUF value type: {value_type}"
        ))),
    }
}

/// Read a GGUF value of a known type (for array elements).
fn read_gguf_value_of_type<R: Read>(r: &mut R, type_id: u32) -> Result<GgufValue> {
    match type_id {
        0 => { let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; Ok(GgufValue::Uint8(buf[0])) }
        1 => { let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; Ok(GgufValue::Int8(buf[0] as i8)) }
        2 => { let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; Ok(GgufValue::Uint16(u16::from_le_bytes(buf))) }
        3 => { let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; Ok(GgufValue::Int16(i16::from_le_bytes(buf))) }
        4 => { let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(GgufValue::Uint32(u32::from_le_bytes(buf))) }
        5 => { let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(GgufValue::Int32(i32::from_le_bytes(buf))) }
        6 => { let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(GgufValue::Float32(f32::from_le_bytes(buf))) }
        7 => { let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; Ok(GgufValue::Bool(buf[0] != 0)) }
        8 => { Ok(GgufValue::String(read_gguf_string(r)?)) }
        10 => { let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(GgufValue::Uint64(u64::from_le_bytes(buf))) }
        11 => { let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(GgufValue::Int64(i64::from_le_bytes(buf))) }
        12 => { let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(GgufValue::Float64(f64::from_le_bytes(buf))) }
        _ => Err(OneBitError::Config(format!("Unknown GGUF array element type: {type_id}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal valid GGUF header with given metadata.
    fn build_gguf_bytes(metadata: &[(&str, u32, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Metadata KV count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        for &(key, value_type, value_bytes) in metadata {
            // Key string
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value type
            buf.extend_from_slice(&value_type.to_le_bytes());
            // Value data
            buf.extend_from_slice(value_bytes);
        }

        buf
    }

    #[test]
    fn test_gguf_header_basic() {
        let data = build_gguf_bytes(&[
            ("general.name", 8, {  // type 8 = STRING
                let s = "test-model";
                let mut v = Vec::new();
                v.extend_from_slice(&(s.len() as u64).to_le_bytes());
                v.extend_from_slice(s.as_bytes());
                // Leak to get a static slice — fine for testing
                Box::leak(v.into_boxed_slice())
            }),
            ("general.layers", 4, &42u32.to_le_bytes()),  // type 4 = UINT32
        ]);

        let header = GgufHeader::read(Cursor::new(&data)).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 0);
        assert_eq!(header.metadata.len(), 2);
        assert_eq!(header.get_str("general.name"), Some("test-model"));
        assert_eq!(header.get_u32("general.layers"), Some(42));
    }

    #[test]
    fn test_gguf_invalid_magic() {
        let data = b"\x00\x00\x00\x00";
        assert!(GgufHeader::read(Cursor::new(data)).is_err());
    }

    #[test]
    fn test_gguf_value_types() {
        let data = build_gguf_bytes(&[
            ("bool_key", 7, &[1u8]),           // BOOL
            ("float_key", 6, &1.5f32.to_le_bytes()),  // FLOAT32
        ]);

        let header = GgufHeader::read(Cursor::new(&data)).unwrap();
        match header.get("bool_key").unwrap() {
            GgufValue::Bool(v) => assert!(*v),
            _ => panic!("Expected bool"),
        }
        assert_eq!(header.get("float_key").unwrap().as_f32(), Some(1.5));
    }
}
