use ndarray::{Array, Array2, IxDyn};

use crate::Result;
use crate::error::OneBitError;
use super::embedding::Embedding;

/// Rotary Position Embedding (RoPE).
///
/// Applies rotation to pairs of dimensions based on position,
/// enabling relative position awareness in attention.
pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    _base: f32,
    /// Precomputed cos table: (max_seq_len, dim/2).
    cos_cache: Array2<f32>,
    /// Precomputed sin table: (max_seq_len, dim/2).
    sin_cache: Array2<f32>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        let half_dim = dim / 2;
        let mut cos_cache = Array2::zeros((max_seq_len, half_dim));
        let mut sin_cache = Array2::zeros((max_seq_len, half_dim));

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
                let angle = pos as f32 * freq;
                cos_cache[[pos, i]] = angle.cos();
                sin_cache[[pos, i]] = angle.sin();
            }
        }

        Self {
            dim,
            max_seq_len,
            _base: base,
            cos_cache,
            sin_cache,
        }
    }

    /// Apply rotary embedding to a tensor.
    ///
    /// `x` shape: (batch, num_heads, seq_len, head_dim)
    /// `position_ids`: sequence positions for each token.
    pub fn apply(
        &self,
        x: &Array<f32, IxDyn>,
        position_ids: &[usize],
    ) -> Result<Array<f32, IxDyn>> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(OneBitError::TensorOp(
                "RoPE expects 4D input (batch, heads, seq, dim)".into(),
            ));
        }

        let batch = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        if head_dim != self.dim {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![self.dim],
                got: vec![head_dim],
            });
        }

        let mut output = x.clone();

        for b in 0..batch {
            for h in 0..num_heads {
                for (s, &pos) in position_ids.iter().enumerate().take(seq_len) {
                    if pos >= self.max_seq_len {
                        return Err(OneBitError::TensorOp(format!(
                            "Position {pos} exceeds max_seq_len {}",
                            self.max_seq_len
                        )));
                    }

                    for i in 0..half_dim {
                        let x0 = x[[b, h, s, i]];
                        let x1 = x[[b, h, s, i + half_dim]];
                        let cos = self.cos_cache[[pos, i]];
                        let sin = self.sin_cache[[pos, i]];

                        output[[b, h, s, i]] = x0 * cos - x1 * sin;
                        output[[b, h, s, i + half_dim]] = x1 * cos + x0 * sin;
                    }
                }
            }
        }

        Ok(output)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// ALiBi (Attention with Linear Biases).
///
/// Adds position-dependent linear biases to attention scores
/// instead of using position embeddings in the input.
pub struct AlibiEmbedding {
    num_heads: usize,
    /// Precomputed slopes per head.
    slopes: Vec<f32>,
}

impl AlibiEmbedding {
    pub fn new(num_heads: usize) -> Self {
        // Slopes follow the geometric sequence from the ALiBi paper
        let slopes = Self::compute_slopes(num_heads);
        Self { num_heads, slopes }
    }

    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        let mut slopes = Vec::with_capacity(num_heads);
        let ratio = 2.0f32.powf(-(8.0 / num_heads as f32));
        for i in 0..num_heads {
            slopes.push(ratio.powi(i as i32 + 1));
        }
        slopes
    }

    /// Compute ALiBi bias matrix for given sequence length.
    ///
    /// Returns shape (num_heads, seq_len, seq_len) with causal mask.
    pub fn bias_matrix(&self, seq_len: usize) -> Array<f32, IxDyn> {
        let mut bias = Array::zeros(IxDyn(&[self.num_heads, seq_len, seq_len]));

        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j <= i {
                        // Causal: only attend to past positions
                        let distance = (i as f32 - j as f32).abs();
                        bias[[h, i, j]] = -slope * distance;
                    } else {
                        bias[[h, i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        bias
    }
}

/// Learned absolute positional embedding.
pub struct LearnedPositionalEmbedding {
    embedding: Embedding,
    max_seq_len: usize,
}

impl LearnedPositionalEmbedding {
    pub fn new(max_seq_len: usize, embed_dim: usize) -> Self {
        Self {
            embedding: Embedding::new(max_seq_len, embed_dim),
            max_seq_len,
        }
    }

    /// Get position embeddings for a given sequence length.
    ///
    /// Returns shape (seq_len, embed_dim).
    pub fn forward(&self, seq_len: usize) -> Result<Array<f32, IxDyn>> {
        if seq_len > self.max_seq_len {
            return Err(OneBitError::TensorOp(format!(
                "Sequence length {seq_len} exceeds max {}", self.max_seq_len
            )));
        }
        let ids: Vec<u32> = (0..seq_len as u32).collect();
        self.embedding.forward_ids(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0);
        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.max_seq_len(), 2048);
    }

    #[test]
    fn test_rope_apply_shape() {
        let rope = RotaryEmbedding::new(8, 128, 10000.0);
        let x = Array::from_elem(IxDyn(&[1, 2, 4, 8]), 1.0f32);
        let positions: Vec<usize> = (0..4).collect();

        let result = rope.apply(&x, &positions).unwrap();
        assert_eq!(result.shape(), &[1, 2, 4, 8]);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE should approximately preserve the L2 norm of each position vector
        let rope = RotaryEmbedding::new(8, 128, 10000.0);
        let x = Array::from_elem(IxDyn(&[1, 1, 1, 8]), 1.0f32);
        let positions = vec![0];

        let result = rope.apply(&x, &positions).unwrap();

        let input_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let output_norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (input_norm - output_norm).abs() < 0.01,
            "Input norm {input_norm}, output norm {output_norm}"
        );
    }

    #[test]
    fn test_alibi_bias_shape() {
        let alibi = AlibiEmbedding::new(4);
        let bias = alibi.bias_matrix(8);
        assert_eq!(bias.shape(), &[4, 8, 8]);
    }

    #[test]
    fn test_alibi_causal() {
        let alibi = AlibiEmbedding::new(2);
        let bias = alibi.bias_matrix(4);

        // Upper triangle (future positions) should be -inf
        for h in 0..2 {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    assert!(bias[[h, i, j]].is_infinite());
                }
            }
        }

        // Diagonal should be 0
        for h in 0..2 {
            for i in 0..4 {
                assert!((bias[[h, i, i]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_alibi_linear_decrease() {
        let alibi = AlibiEmbedding::new(1);
        let bias = alibi.bias_matrix(4);

        // For head 0, bias[0, 3, 0] should be more negative than bias[0, 2, 0]
        assert!(bias[[0, 3, 0]] < bias[[0, 2, 0]]);
        assert!(bias[[0, 2, 0]] < bias[[0, 1, 0]]);
    }

    #[test]
    fn test_learned_positional() {
        let pos_emb = LearnedPositionalEmbedding::new(128, 32);
        let result = pos_emb.forward(10).unwrap();
        assert_eq!(result.shape(), &[10, 32]);
    }

    #[test]
    fn test_learned_positional_exceeds_max() {
        let pos_emb = LearnedPositionalEmbedding::new(10, 32);
        assert!(pos_emb.forward(11).is_err());
    }
}
