//! ONNX export stub.
//!
//! Provides a placeholder for future ONNX export support.
//! ONNX export requires generating a protobuf-encoded graph, which
//! is deferred to a future phase when the `onnx` feature is implemented.

use crate::error::OneBitError;
use crate::nn::Parameter;
use crate::Result;

/// Placeholder for ONNX export configuration.
#[derive(Debug, Clone)]
pub struct OnnxExportConfig {
    /// ONNX opset version.
    pub opset_version: i64,
    /// Model name in the ONNX graph.
    pub model_name: String,
}

impl Default for OnnxExportConfig {
    fn default() -> Self {
        Self {
            opset_version: 17,
            model_name: "onebitllm_model".into(),
        }
    }
}

/// Export model parameters to ONNX format.
///
/// **Not yet implemented.** Returns an error explaining the feature is not available.
pub fn export_onnx(_params: &[&Parameter], _config: &OnnxExportConfig) -> Result<Vec<u8>> {
    Err(OneBitError::Other(
        "ONNX export is not yet implemented. Use SafeTensors or OBM format instead.".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_export_not_implemented() {
        let config = OnnxExportConfig::default();
        assert_eq!(config.opset_version, 17);
        assert!(export_onnx(&[], &config).is_err());
    }
}
