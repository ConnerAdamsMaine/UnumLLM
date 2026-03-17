use ndarray::{Array, IxDyn};

use crate::Result;
use crate::error::OneBitError;

/// KV cache for a single attention layer.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    /// Cached keys: (batch, num_kv_heads, cached_seq_len, head_dim).
    pub keys: Array<f32, IxDyn>,
    /// Cached values: (batch, num_kv_heads, cached_seq_len, head_dim).
    pub values: Array<f32, IxDyn>,
    /// Current number of cached positions.
    pub seq_len: usize,
    /// Maximum sequence length this cache can hold.
    max_seq_len: usize,
}

impl LayerKvCache {
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            keys: Array::zeros(IxDyn(&[batch_size, num_kv_heads, max_seq_len, head_dim])),
            values: Array::zeros(IxDyn(&[batch_size, num_kv_heads, max_seq_len, head_dim])),
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Append new keys/values to this layer's cache.
    ///
    /// `new_keys` and `new_values` shape: (batch, num_kv_heads, new_len, head_dim).
    pub fn update(
        &mut self,
        new_keys: &Array<f32, IxDyn>,
        new_values: &Array<f32, IxDyn>,
    ) -> Result<()> {
        let new_len = new_keys.shape()[2];
        let end = self.seq_len + new_len;

        if end > self.max_seq_len {
            return Err(OneBitError::Inference(format!(
                "KV cache overflow: {end} exceeds max_seq_len {}",
                self.max_seq_len
            )));
        }

        let batch = self.keys.shape()[0];
        let num_heads = self.keys.shape()[1];
        let head_dim = self.keys.shape()[3];

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..new_len {
                    for d in 0..head_dim {
                        self.keys[[b, h, self.seq_len + s, d]] = new_keys[[b, h, s, d]];
                        self.values[[b, h, self.seq_len + s, d]] = new_values[[b, h, s, d]];
                    }
                }
            }
        }

        self.seq_len = end;
        Ok(())
    }

    /// Get cached keys up to current seq_len: (batch, heads, seq_len, dim).
    pub fn cached_keys(&self) -> Array<f32, IxDyn> {
        let shape = self.keys.shape();
        let batch = shape[0];
        let heads = shape[1];
        let dim = shape[3];

        let mut out = Array::zeros(IxDyn(&[batch, heads, self.seq_len, dim]));
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..self.seq_len {
                    for d in 0..dim {
                        out[[b, h, s, d]] = self.keys[[b, h, s, d]];
                    }
                }
            }
        }
        out
    }

    /// Get cached values up to current seq_len.
    pub fn cached_values(&self) -> Array<f32, IxDyn> {
        let shape = self.values.shape();
        let batch = shape[0];
        let heads = shape[1];
        let dim = shape[3];

        let mut out = Array::zeros(IxDyn(&[batch, heads, self.seq_len, dim]));
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..self.seq_len {
                    for d in 0..dim {
                        out[[b, h, s, d]] = self.values[[b, h, s, d]];
                    }
                }
            }
        }
        out
    }

    /// Reset cache to empty.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.keys.fill(0.0);
        self.values.fill(0.0);
    }
}

/// KV cache for all layers in a model.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(batch_size, num_kv_heads, head_dim, max_seq_len))
            .collect();
        Self {
            layers,
            max_seq_len,
        }
    }

    /// Update a specific layer's cache.
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: &Array<f32, IxDyn>,
        new_values: &Array<f32, IxDyn>,
    ) -> Result<()> {
        if layer_idx >= self.layers.len() {
            return Err(OneBitError::Inference(format!(
                "Layer index {layer_idx} out of bounds (num_layers = {})",
                self.layers.len()
            )));
        }
        self.layers[layer_idx].update(new_keys, new_values)
    }

    /// Get the cache for a specific layer.
    pub fn get(&self, layer_idx: usize) -> Option<&LayerKvCache> {
        self.layers.get(layer_idx)
    }

    /// Get mutable cache for a specific layer.
    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut LayerKvCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Reset all caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Current cached sequence length (same across all layers).
    pub fn cached_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache_new() {
        let cache = LayerKvCache::new(1, 4, 8, 128);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.keys.shape(), &[1, 4, 128, 8]);
    }

    #[test]
    fn test_layer_cache_update() {
        let mut cache = LayerKvCache::new(1, 2, 4, 16);
        let new_k = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 1.0f32);
        let new_v = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 2.0f32);

        cache.update(&new_k, &new_v).unwrap();
        assert_eq!(cache.seq_len, 3);

        // Append more
        let new_k2 = Array::from_elem(IxDyn(&[1, 2, 2, 4]), 3.0f32);
        let new_v2 = Array::from_elem(IxDyn(&[1, 2, 2, 4]), 4.0f32);
        cache.update(&new_k2, &new_v2).unwrap();
        assert_eq!(cache.seq_len, 5);
    }

    #[test]
    fn test_layer_cache_overflow() {
        let mut cache = LayerKvCache::new(1, 2, 4, 4);
        let new_k = Array::from_elem(IxDyn(&[1, 2, 5, 4]), 1.0f32);
        let new_v = Array::from_elem(IxDyn(&[1, 2, 5, 4]), 1.0f32);
        assert!(cache.update(&new_k, &new_v).is_err());
    }

    #[test]
    fn test_layer_cache_get_cached() {
        let mut cache = LayerKvCache::new(1, 2, 4, 16);
        let new_k = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 5.0f32);
        let new_v = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 7.0f32);
        cache.update(&new_k, &new_v).unwrap();

        let k = cache.cached_keys();
        assert_eq!(k.shape(), &[1, 2, 3, 4]);
        assert_eq!(k[[0, 0, 0, 0]], 5.0);

        let v = cache.cached_values();
        assert_eq!(v.shape(), &[1, 2, 3, 4]);
        assert_eq!(v[[0, 0, 0, 0]], 7.0);
    }

    #[test]
    fn test_layer_cache_clear() {
        let mut cache = LayerKvCache::new(1, 2, 4, 16);
        let new_k = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 1.0f32);
        let new_v = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 1.0f32);
        cache.update(&new_k, &new_v).unwrap();
        assert_eq!(cache.seq_len, 3);

        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_kv_cache_multi_layer() {
        let mut cache = KvCache::new(4, 1, 2, 8, 128);
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.cached_len(), 0);

        let k = Array::from_elem(IxDyn(&[1, 2, 5, 8]), 1.0f32);
        let v = Array::from_elem(IxDyn(&[1, 2, 5, 8]), 1.0f32);

        cache.update(0, &k, &v).unwrap();
        cache.update(1, &k, &v).unwrap();

        assert_eq!(cache.get(0).unwrap().seq_len, 5);
        assert_eq!(cache.get(1).unwrap().seq_len, 5);
        assert_eq!(cache.get(2).unwrap().seq_len, 0);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(2, 1, 2, 8, 128);
        let k = Array::from_elem(IxDyn(&[1, 2, 5, 8]), 1.0f32);
        let v = Array::from_elem(IxDyn(&[1, 2, 5, 8]), 1.0f32);
        cache.update(0, &k, &v).unwrap();
        cache.update(1, &k, &v).unwrap();

        cache.clear();
        assert_eq!(cache.cached_len(), 0);
        assert_eq!(cache.get(0).unwrap().seq_len, 0);
    }

    #[test]
    fn test_kv_cache_invalid_layer() {
        let mut cache = KvCache::new(2, 1, 2, 8, 128);
        let k = Array::from_elem(IxDyn(&[1, 2, 1, 8]), 1.0f32);
        let v = Array::from_elem(IxDyn(&[1, 2, 1, 8]), 1.0f32);
        assert!(cache.update(5, &k, &v).is_err());
    }
}
