use ndarray::{s, Array, Ix2, IxDyn};

use super::linear::QuantizedLinear;
use super::module::{Module, Parameter};
use crate::error::OneBitError;
use crate::quant::QuantConfig;
use crate::Result;

/// Configuration for attention layers.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    /// None = multi-head attention, Some(n) = grouped-query attention.
    pub num_kv_heads: Option<usize>,
    pub head_dim: usize,
    pub use_bias: bool,
    pub quant_config: QuantConfig,
}

impl AttentionConfig {
    /// Number of KV heads (defaults to num_heads for MHA).
    pub fn kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }
}

/// Multi-Head / Grouped-Query Attention with quantized linear projections.
pub struct Attention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    config: AttentionConfig,
}

impl Attention {
    pub fn new(config: AttentionConfig) -> Self {
        let kv_heads = config.kv_heads();
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = kv_heads * config.head_dim;

        Self {
            q_proj: QuantizedLinear::new(
                config.embed_dim,
                q_dim,
                config.use_bias,
                config.quant_config.clone(),
            ),
            k_proj: QuantizedLinear::new(
                config.embed_dim,
                kv_dim,
                config.use_bias,
                config.quant_config.clone(),
            ),
            v_proj: QuantizedLinear::new(
                config.embed_dim,
                kv_dim,
                config.use_bias,
                config.quant_config.clone(),
            ),
            o_proj: QuantizedLinear::new(
                q_dim,
                config.embed_dim,
                config.use_bias,
                config.quant_config.clone(),
            ),
            config,
        }
    }

    /// Compute scaled dot-product attention.
    ///
    /// `q`, `k`, `v`: (batch, num_heads, seq_len, head_dim)
    /// `mask`: optional (batch, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
    fn scaled_dot_product_attention(
        q: &Array<f32, IxDyn>,
        k: &Array<f32, IxDyn>,
        v: &Array<f32, IxDyn>,
        mask: Option<&Array<f32, IxDyn>>,
        head_dim: usize,
    ) -> Result<Array<f32, IxDyn>> {
        let batch = q.shape()[0];
        let num_heads = q.shape()[1];
        let q_len = q.shape()[2];
        let kv_len = k.shape()[2];
        let scale = (head_dim as f32).sqrt();

        // Compute attention scores: Q @ K^T / sqrt(d)
        let mut scores = Array::zeros(IxDyn(&[batch, num_heads, q_len, kv_len]));
        for b in 0..batch {
            for h in 0..num_heads {
                let q_mat = q
                    .slice(s![b, h, .., ..])
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
                let k_mat = k
                    .slice(s![b, h, .., ..])
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
                let mut score_mat = q_mat.dot(&k_mat.t());
                score_mat.mapv_inplace(|v| v / scale);
                scores
                    .slice_mut(s![b, h, .., ..])
                    .assign(&score_mat.into_dyn());
            }
        }

        // Apply mask (typically causal)
        if let Some(mask) = mask {
            for b in 0..batch {
                for h in 0..num_heads {
                    for i in 0..q_len {
                        for j in 0..kv_len {
                            let mask_b = if mask.shape()[0] == 1 { 0 } else { b };
                            let mask_h = if mask.shape()[1] == 1 { 0 } else { h };
                            scores[[b, h, i, j]] += mask[[mask_b, mask_h, i, j]];
                        }
                    }
                }
            }
        }

        // Softmax over kv_len dimension
        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..q_len {
                    let max_val = (0..kv_len)
                        .map(|j| scores[[b, h, i, j]])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for j in 0..kv_len {
                        scores[[b, h, i, j]] = (scores[[b, h, i, j]] - max_val).exp();
                        sum += scores[[b, h, i, j]];
                    }
                    for j in 0..kv_len {
                        scores[[b, h, i, j]] /= sum;
                    }
                }
            }
        }

        // Multiply by V: scores @ V
        let mut output = Array::zeros(IxDyn(&[batch, num_heads, q_len, head_dim]));
        for b in 0..batch {
            for h in 0..num_heads {
                let score_mat = scores
                    .slice(s![b, h, .., ..])
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
                let v_mat = v
                    .slice(s![b, h, .., ..])
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
                let out_mat = score_mat.dot(&v_mat);
                output
                    .slice_mut(s![b, h, .., ..])
                    .assign(&out_mat.into_dyn());
            }
        }

        Ok(output)
    }

    /// Create a causal attention mask.
    pub fn causal_mask(seq_len: usize) -> Array<f32, IxDyn> {
        let mut mask = Array::from_elem(IxDyn(&[1, 1, seq_len, seq_len]), 0.0f32);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[[0, 0, i, j]] = f32::NEG_INFINITY;
            }
        }
        mask
    }
}

impl Module for Attention {
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        // input: (batch, seq_len, embed_dim)
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(OneBitError::TensorOp(
                "Attention expects 3D input (batch, seq, embed)".into(),
            ));
        }

        let batch = shape[0];
        let seq_len = shape[1];
        let num_heads = self.config.num_heads;
        let kv_heads = self.config.kv_heads();
        let head_dim = self.config.head_dim;

        // Project Q, K, V
        let q = self.q_proj.forward_inference(input)?;
        let k = self.k_proj.forward_inference(input)?;
        let v = self.v_proj.forward_inference(input)?;

        // Reshape to (batch, seq_len, num_heads, head_dim) then transpose to (batch, num_heads, seq_len, head_dim)
        let q = transpose_to_heads(&q, batch, seq_len, num_heads, head_dim)?;
        let k = transpose_to_heads(&k, batch, seq_len, kv_heads, head_dim)?;
        let v = transpose_to_heads(&v, batch, seq_len, kv_heads, head_dim)?;

        // Handle GQA: repeat KV heads to match query heads
        let (k, v) = if kv_heads < num_heads {
            let repeat = num_heads / kv_heads;
            let k_expanded = repeat_kv(&k, repeat)?;
            let v_expanded = repeat_kv(&v, repeat)?;
            (k_expanded, v_expanded)
        } else {
            (k, v)
        };

        // Causal mask
        let mask = Self::causal_mask(seq_len);

        // Scaled dot-product attention
        let attn_output = Self::scaled_dot_product_attention(&q, &k, &v, Some(&mask), head_dim)?;

        // Transpose back: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads * head_dim)
        let attn_output = transpose_from_heads(&attn_output, batch, seq_len, num_heads, head_dim)?;

        // Output projection
        self.o_proj.forward_inference(&attn_output.into_dyn())
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters_mut());
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.o_proj.parameters_mut());
        params
    }

    fn name(&self) -> &str {
        "Attention"
    }
}

/// Transpose from (batch, seq, heads, dim) to (batch, heads, seq, dim) with contiguous layout.
fn transpose_to_heads(
    x: &Array<f32, IxDyn>,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Array<f32, IxDyn>> {
    let reshaped = x
        .clone()
        .into_shape_with_order(IxDyn(&[batch, seq_len, num_heads, head_dim]))
        .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
    let permuted = reshaped.permuted_axes(IxDyn(&[0, 2, 1, 3]));
    Array::from_shape_vec(
        IxDyn(&[batch, num_heads, seq_len, head_dim]),
        permuted.iter().copied().collect(),
    )
    .map_err(|e| OneBitError::TensorOp(e.to_string()))
}

/// Transpose from (batch, heads, seq, dim) to (batch, seq, heads*dim) with contiguous layout.
fn transpose_from_heads(
    x: &Array<f32, IxDyn>,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Array<f32, IxDyn>> {
    let reshaped = x
        .clone()
        .into_shape_with_order(IxDyn(&[batch, num_heads, seq_len, head_dim]))
        .map_err(|e| OneBitError::TensorOp(e.to_string()))?;
    let permuted = reshaped.permuted_axes(IxDyn(&[0, 2, 1, 3]));
    Array::from_shape_vec(
        IxDyn(&[batch, seq_len, num_heads * head_dim]),
        permuted.iter().copied().collect(),
    )
    .map_err(|e| OneBitError::TensorOp(e.to_string()))
}

/// Repeat KV heads to match the number of query heads (for GQA).
fn repeat_kv(x: &Array<f32, IxDyn>, repeat: usize) -> Result<Array<f32, IxDyn>> {
    if repeat == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape();
    let batch = shape[0];
    let kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];
    let new_heads = kv_heads * repeat;

    let mut expanded = Array::zeros(IxDyn(&[batch, new_heads, seq_len, head_dim]));
    for b in 0..batch {
        for kv_h in 0..kv_heads {
            for r in 0..repeat {
                let new_h = kv_h * repeat + r;
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        expanded[[b, new_h, s, d]] = x[[b, kv_h, s, d]];
                    }
                }
            }
        }
    }

    Ok(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(num_heads: usize, kv_heads: Option<usize>) -> AttentionConfig {
        let head_dim = 8;
        AttentionConfig {
            embed_dim: num_heads * head_dim,
            num_heads,
            num_kv_heads: kv_heads,
            head_dim,
            use_bias: false,
            quant_config: QuantConfig::per_tensor(),
        }
    }

    #[test]
    fn test_mha_output_shape() {
        let attn = Attention::new(test_config(4, None));
        let input = Array::from_elem(IxDyn(&[1, 8, 32]), 0.1f32);
        let output = attn.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[1, 8, 32]);
    }

    #[test]
    fn test_gqa_output_shape() {
        // 4 query heads, 2 kv heads
        let attn = Attention::new(test_config(4, Some(2)));
        let input = Array::from_elem(IxDyn(&[1, 8, 32]), 0.1f32);
        let output = attn.forward_inference(&input).unwrap();
        assert_eq!(output.shape(), &[1, 8, 32]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = Attention::causal_mask(4);
        assert_eq!(mask.shape(), &[1, 1, 4, 4]);

        // Lower triangle should be 0
        assert_eq!(mask[[0, 0, 0, 0]], 0.0);
        assert_eq!(mask[[0, 0, 1, 0]], 0.0);
        assert_eq!(mask[[0, 0, 1, 1]], 0.0);

        // Upper triangle should be -inf
        assert!(mask[[0, 0, 0, 1]].is_infinite());
        assert!(mask[[0, 0, 0, 2]].is_infinite());
    }

    #[test]
    fn test_attention_parameters() {
        let attn = Attention::new(test_config(4, None));
        // 4 projections: Q, K, V, O, each with weight only (no bias)
        // Q: (32, 32), K: (32, 32), V: (32, 32), O: (32, 32)
        assert_eq!(attn.num_parameters(), 4 * 32 * 32);
    }

    #[test]
    fn test_repeat_kv() {
        let x = Array::from_elem(IxDyn(&[1, 2, 4, 8]), 1.0f32);
        let expanded = repeat_kv(&x, 3).unwrap();
        assert_eq!(expanded.shape(), &[1, 6, 4, 8]);
    }
}
