use ndarray::{Array, IxDyn};
use rand::Rng;

use super::module::{Module, Parameter};
use crate::error::OneBitError;
use crate::Result;

/// Token embedding layer.
///
/// Stores a lookup table of shape (vocab_size, embed_dim).
/// Embeddings are kept in full precision (not quantized) since they
/// are sparsely accessed and quantization would degrade quality.
pub struct Embedding {
    weight: Parameter,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    /// Create a new embedding with random initialization.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..vocab_size * embed_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let weight = Parameter::new(
            "embedding",
            Array::from_shape_vec(IxDyn(&[vocab_size, embed_dim]), data).unwrap(),
        );

        Self {
            weight,
            vocab_size,
            embed_dim,
        }
    }

    /// Create from existing weight data.
    pub fn from_weights(weight: Array<f32, IxDyn>) -> Result<Self> {
        if weight.ndim() != 2 {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![0, 0],
                got: weight.shape().to_vec(),
            });
        }
        let vocab_size = weight.shape()[0];
        let embed_dim = weight.shape()[1];

        Ok(Self {
            weight: Parameter::new("embedding", weight),
            vocab_size,
            embed_dim,
        })
    }

    /// Look up embeddings for token IDs.
    ///
    /// `ids` can be any shape. Returns shape (*ids_shape, embed_dim).
    pub fn forward_ids(&self, ids: &[u32]) -> Result<Array<f32, IxDyn>> {
        let mut output = Vec::with_capacity(ids.len() * self.embed_dim);

        for &id in ids {
            let idx = id as usize;
            if idx >= self.vocab_size {
                return Err(OneBitError::TensorOp(format!(
                    "Token ID {id} out of bounds (vocab_size = {})",
                    self.vocab_size
                )));
            }
            let row_start = idx * self.embed_dim;
            let weight_slice = self.weight.data.as_slice().unwrap();
            output.extend_from_slice(&weight_slice[row_start..row_start + self.embed_dim]);
        }

        Ok(Array::from_shape_vec(IxDyn(&[ids.len(), self.embed_dim]), output).unwrap())
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

impl Module for Embedding {
    fn forward_inference(&self, input: &Array<f32, IxDyn>) -> Result<Array<f32, IxDyn>> {
        // Input is expected to be token IDs cast to f32
        let ids: Vec<u32> = input.iter().map(|&x| x as u32).collect();
        let input_shape = input.shape().to_vec();

        let embeddings = self.forward_ids(&ids)?;

        // Reshape to (*input_shape, embed_dim)
        let mut out_shape = input_shape;
        out_shape.push(self.embed_dim);
        embeddings
            .into_shape_with_order(IxDyn(&out_shape))
            .map_err(|e| OneBitError::TensorOp(e.to_string()))
    }

    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_embedding_creation() {
        let emb = Embedding::new(100, 32);
        assert_eq!(emb.vocab_size(), 100);
        assert_eq!(emb.embed_dim(), 32);
        assert_eq!(emb.num_parameters(), 100 * 32);
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = Embedding::new(10, 4);
        let result = emb.forward_ids(&[0, 1, 2]).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_embedding_forward_inference() {
        let emb = Embedding::new(10, 4);
        let input = array![0.0f32, 1.0, 2.0].into_dyn();
        let result = emb.forward_inference(&input).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
    }

    #[test]
    fn test_embedding_2d_input() {
        let emb = Embedding::new(10, 4);
        let input =
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let result = emb.forward_inference(&input).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_embedding_out_of_bounds() {
        let emb = Embedding::new(10, 4);
        assert!(emb.forward_ids(&[10]).is_err());
    }

    #[test]
    fn test_embedding_deterministic() {
        let emb = Embedding::new(10, 4);
        let r1 = emb.forward_ids(&[3]).unwrap();
        let r2 = emb.forward_ids(&[3]).unwrap();
        assert_eq!(r1, r2);
    }
}
