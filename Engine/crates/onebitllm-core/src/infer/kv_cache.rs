use ndarray::{Array, IxDyn};

use crate::error::OneBitError;
use crate::Result;

const INT4_MIN: i8 = -8;
const INT4_MAX: i8 = 7;
const INT4_SCALE_DENOM: f32 = 7.0;
const ZERO_NIBBLE: u8 = 8;

#[derive(Debug, Clone)]
struct QuantizedCacheTensor {
    data: Vec<u8>,
    scales: Vec<f32>,
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl QuantizedCacheTensor {
    fn new(batch_size: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let total_elements = batch_size * num_heads * max_seq_len * head_dim;
        let total_vectors = batch_size * num_heads * max_seq_len;
        Self {
            data: vec![0u8; total_elements.div_ceil(2)],
            scales: vec![0.0; total_vectors],
            batch_size,
            num_heads,
            head_dim,
            max_seq_len,
        }
    }

    fn update_range(
        &mut self,
        seq_offset: usize,
        source: &Array<f32, IxDyn>,
        new_len: usize,
    ) -> Result<()> {
        self.validate_source_shape(source, new_len)?;

        for b in 0..self.batch_size {
            for h in 0..self.num_heads {
                for s in 0..new_len {
                    let cache_seq = seq_offset + s;
                    let mut max_abs = 0.0f32;
                    for d in 0..self.head_dim {
                        max_abs = max_abs.max(source[[b, h, s, d]].abs());
                    }

                    let scale = if max_abs <= f32::EPSILON {
                        0.0
                    } else {
                        max_abs / INT4_SCALE_DENOM
                    };
                    let scale_index = self.vector_index(b, h, cache_seq);
                    self.scales[scale_index] = scale;

                    for d in 0..self.head_dim {
                        let quantized = if scale == 0.0 {
                            0
                        } else {
                            (source[[b, h, s, d]] / scale)
                                .round()
                                .clamp(INT4_MIN as f32, INT4_MAX as f32)
                                as i8
                        };
                        self.set_quantized(b, h, cache_seq, d, quantized);
                    }
                }
            }
        }

        Ok(())
    }

    fn dequantize_prefix(&self, seq_len: usize) -> Array<f32, IxDyn> {
        let mut out = Array::zeros(IxDyn(&[
            self.batch_size,
            self.num_heads,
            seq_len,
            self.head_dim,
        ]));

        for b in 0..self.batch_size {
            for h in 0..self.num_heads {
                for s in 0..seq_len {
                    let scale = self.scales[self.vector_index(b, h, s)];
                    for d in 0..self.head_dim {
                        out[[b, h, s, d]] = self.get_quantized(b, h, s, d) as f32 * scale;
                    }
                }
            }
        }

        out
    }

    fn clear(&mut self) {
        self.data.fill(0);
        self.scales.fill(0.0);
    }

    fn storage_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    fn validate_source_shape(&self, source: &Array<f32, IxDyn>, new_len: usize) -> Result<()> {
        let expected = [self.batch_size, self.num_heads, new_len, self.head_dim];
        if source.shape() != expected {
            return Err(OneBitError::ShapeMismatch {
                expected: expected.to_vec(),
                got: source.shape().to_vec(),
            });
        }
        Ok(())
    }

    fn vector_index(&self, batch: usize, head: usize, seq: usize) -> usize {
        ((batch * self.num_heads + head) * self.max_seq_len) + seq
    }

    fn element_index(&self, batch: usize, head: usize, seq: usize, dim: usize) -> usize {
        ((((batch * self.num_heads) + head) * self.max_seq_len + seq) * self.head_dim) + dim
    }

    fn set_quantized(&mut self, batch: usize, head: usize, seq: usize, dim: usize, value: i8) {
        let encoded = (value + ZERO_NIBBLE as i8) as u8;
        let flat_index = self.element_index(batch, head, seq, dim);
        let byte_index = flat_index / 2;
        let shift = if flat_index % 2 == 0 { 0 } else { 4 };
        self.data[byte_index] &= !(0x0Fu8 << shift);
        self.data[byte_index] |= encoded << shift;
    }

    fn get_quantized(&self, batch: usize, head: usize, seq: usize, dim: usize) -> i8 {
        let flat_index = self.element_index(batch, head, seq, dim);
        let byte_index = flat_index / 2;
        let shift = if flat_index % 2 == 0 { 0 } else { 4 };
        let encoded = (self.data[byte_index] >> shift) & 0x0F;
        encoded as i8 - ZERO_NIBBLE as i8
    }
}

/// 4-bit KV cache for a single attention layer.
///
/// Keys and values are stored as packed signed int4 values with a per
/// `(batch, kv_head, seq)` scale factor so reconstruction stays local and
/// cheap while cutting memory usage substantially relative to dense `f32`.
#[derive(Debug, Clone)]
pub struct LayerKvCache {
    keys: QuantizedCacheTensor,
    values: QuantizedCacheTensor,
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
            keys: QuantizedCacheTensor::new(batch_size, num_kv_heads, head_dim, max_seq_len),
            values: QuantizedCacheTensor::new(batch_size, num_kv_heads, head_dim, max_seq_len),
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
        if new_keys.shape() != new_values.shape() {
            return Err(OneBitError::ShapeMismatch {
                expected: new_keys.shape().to_vec(),
                got: new_values.shape().to_vec(),
            });
        }
        if new_keys.ndim() != 4 {
            return Err(OneBitError::Inference(format!(
                "KV cache update expects 4D tensors, got {}D",
                new_keys.ndim()
            )));
        }

        let new_len = new_keys.shape()[2];
        let end = self.seq_len + new_len;
        if end > self.max_seq_len {
            return Err(OneBitError::Inference(format!(
                "KV cache overflow: {end} exceeds max_seq_len {}",
                self.max_seq_len
            )));
        }

        self.keys.update_range(self.seq_len, new_keys, new_len)?;
        self.values
            .update_range(self.seq_len, new_values, new_len)?;
        self.seq_len = end;
        Ok(())
    }

    /// Get cached keys up to current seq_len: (batch, heads, seq_len, dim).
    pub fn cached_keys(&self) -> Array<f32, IxDyn> {
        self.keys.dequantize_prefix(self.seq_len)
    }

    /// Get cached values up to current seq_len.
    pub fn cached_values(&self) -> Array<f32, IxDyn> {
        self.values.dequantize_prefix(self.seq_len)
    }

    /// Total storage used by the packed key/value cache, including scales.
    pub fn storage_bytes(&self) -> usize {
        self.keys.storage_bytes() + self.values.storage_bytes()
    }

    /// The dense `f32` storage cost for the same cache shape.
    pub fn dense_f32_storage_bytes(&self) -> usize {
        self.keys.batch_size
            * self.keys.num_heads
            * self.max_seq_len
            * self.keys.head_dim
            * std::mem::size_of::<f32>()
            * 2
    }

    /// Reset cache to empty.
    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.keys.clear();
        self.values.clear();
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
        self.layers.first().map(|layer| layer.seq_len).unwrap_or(0)
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
        assert_eq!(cache.cached_keys().shape(), &[1, 4, 0, 8]);
        assert_eq!(cache.cached_values().shape(), &[1, 4, 0, 8]);
    }

    #[test]
    fn test_layer_cache_update() {
        let mut cache = LayerKvCache::new(1, 2, 4, 16);
        let new_k = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 1.0f32);
        let new_v = Array::from_elem(IxDyn(&[1, 2, 3, 4]), 2.0f32);

        cache.update(&new_k, &new_v).unwrap();
        assert_eq!(cache.seq_len, 3);

        let cached_k = cache.cached_keys();
        let cached_v = cache.cached_values();
        assert_eq!(cached_k.shape(), &[1, 2, 3, 4]);
        assert_eq!(cached_v.shape(), &[1, 2, 3, 4]);

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
        let new_k = Array::from_shape_vec(
            IxDyn(&[1, 2, 3, 4]),
            vec![
                5.0, -2.0, 0.5, 1.5, 4.0, -4.0, 2.0, -1.0, 3.0, 1.0, -2.0, 0.0, 7.0, -7.0, 1.0,
                -1.0, -3.5, 2.0, 1.0, -1.0, 2.5, -2.5, 0.5, -0.5,
            ],
        )
        .unwrap();
        let new_v = &new_k * 0.5;
        cache.update(&new_k, &new_v).unwrap();

        let k = cache.cached_keys();
        let v = cache.cached_values();
        assert_eq!(k.shape(), &[1, 2, 3, 4]);
        assert_eq!(v.shape(), &[1, 2, 3, 4]);

        let k_err = (&k - &new_k).mapv(|value| value.abs());
        let v_err = (&v - &new_v).mapv(|value| value.abs());
        assert!(k_err.iter().all(|err| *err <= 0.6));
        assert!(v_err.iter().all(|err| *err <= 0.35));
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
        assert_eq!(cache.cached_keys().shape(), &[1, 2, 0, 4]);
    }

    #[test]
    fn test_layer_cache_is_more_compact_than_dense_storage() {
        let cache = LayerKvCache::new(1, 8, 64, 256);
        assert!(cache.storage_bytes() < cache.dense_f32_storage_bytes());
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
