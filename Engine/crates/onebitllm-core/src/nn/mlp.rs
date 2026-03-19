use ndarray::{Array, IxDyn};

use super::activation::ActivationFn;
use super::linear::QuantizedLinear;
use super::module::{Module, Parameter};
use crate::quant::QuantConfig;
use crate::Result;

/// MLP block with quantized linear layers.
///
/// Supports both standard (Linear -> Act -> Linear) and
/// SwiGLU (gate_proj + up_proj -> SiLU(gate) * up -> down_proj) variants.
pub struct MlpBlock {
    /// Gate projection (only for SwiGLU): (hidden_dim, intermediate_dim).
    gate_proj: Option<QuantizedLinear>,
    /// Up projection: (hidden_dim, intermediate_dim).
    up_proj: QuantizedLinear,
    /// Down projection: (intermediate_dim, hidden_dim).
    down_proj: QuantizedLinear,
    activation: ActivationFn,
    _hidden_dim: usize,
    _intermediate_dim: usize,
}

impl MlpBlock {
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        activation: ActivationFn,
        quant_config: QuantConfig,
    ) -> Self {
        let use_gate = matches!(activation, ActivationFn::SwiGLU);

        let gate_proj = if use_gate {
            Some(QuantizedLinear::new(
                hidden_dim,
                intermediate_dim,
                false,
                quant_config.clone(),
            ))
        } else {
            None
        };

        let up_proj =
            QuantizedLinear::new(hidden_dim, intermediate_dim, false, quant_config.clone());
        let down_proj = QuantizedLinear::new(intermediate_dim, hidden_dim, false, quant_config);

        Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
            _hidden_dim: hidden_dim,
            _intermediate_dim: intermediate_dim,
        }
    }
}

impl Module for MlpBlock {
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        if let Some(gate_proj) = &self.gate_proj {
            // SwiGLU: SiLU(gate(x)) * up(x)
            let gate = gate_proj.forward_inference(input)?;
            let up = self.up_proj.forward_inference(input)?;
            let activated_gate = self.activation.apply(&gate);
            let hidden = &activated_gate * &up;
            self.down_proj.forward_inference(&hidden)
        } else {
            // Standard: act(up(x)) -> down
            let up = self.up_proj.forward_inference(input)?;
            let activated = self.activation.apply(&up);
            self.down_proj.forward_inference(&activated)
        }
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        if let Some(gate) = &self.gate_proj {
            params.extend(gate.parameters());
        }
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = Vec::new();
        if let Some(gate) = &mut self.gate_proj {
            params.extend(gate.parameters_mut());
        }
        params.extend(self.up_proj.parameters_mut());
        params.extend(self.down_proj.parameters_mut());
        params
    }

    fn name(&self) -> &str {
        "MlpBlock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_mlp_shape() {
        let mlp = MlpBlock::new(32, 64, ActivationFn::GELU, QuantConfig::per_tensor());
        let input = Array::from_elem(IxDyn(&[2, 8, 32]), 0.1f32);
        let output = mlp.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8, 32]);
    }

    #[test]
    fn test_swiglu_mlp_shape() {
        let mlp = MlpBlock::new(32, 64, ActivationFn::SwiGLU, QuantConfig::per_tensor());
        let input = Array::from_elem(IxDyn(&[2, 8, 32]), 0.1f32);
        let output = mlp.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8, 32]);
    }

    #[test]
    fn test_standard_mlp_params() {
        let mlp = MlpBlock::new(32, 64, ActivationFn::GELU, QuantConfig::per_tensor());
        // up: 32*64 + down: 64*32 = 4096
        assert_eq!(mlp.num_parameters(), 32 * 64 + 64 * 32);
    }

    #[test]
    fn test_swiglu_mlp_params() {
        let mlp = MlpBlock::new(32, 64, ActivationFn::SwiGLU, QuantConfig::per_tensor());
        // gate: 32*64 + up: 32*64 + down: 64*32 = 6144
        assert_eq!(mlp.num_parameters(), 32 * 64 * 3);
    }
}
