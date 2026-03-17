use crate::Result;
use crate::error::OneBitError;
use crate::quant::ternary::{TernaryWeight, absmean_quantize};

/// Number of ternary weights packed per u64 word (2 bits each).
const WEIGHTS_PER_WORD: usize = 32;

/// Bitpacked ternary weight storage.
///
/// Each weight occupies 2 bits within u64 words. Weight `i` is stored
/// in bits `[2*(i%32), 2*(i%32)+1]` of word `i/32`.
///
/// Encoding: 0b00 = 0, 0b01 = +1, 0b10 = -1, 0b11 = reserved/invalid.
#[derive(Debug, Clone)]
pub struct PackedTernary {
    data: Vec<u64>,
    len: usize,
}

impl PackedTernary {
    /// Create a new packed storage initialized to zero weights.
    pub fn new(len: usize) -> Self {
        let num_words = (len + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
        Self {
            data: vec![0u64; num_words],
            len,
        }
    }

    /// Pack a slice of ternary weights.
    pub fn from_ternary_slice(weights: &[TernaryWeight]) -> Self {
        let len = weights.len();
        let num_words = (len + WEIGHTS_PER_WORD - 1) / WEIGHTS_PER_WORD;
        let mut data = vec![0u64; num_words];

        for (i, &w) in weights.iter().enumerate() {
            let word_idx = i / WEIGHTS_PER_WORD;
            let bit_idx = (i % WEIGHTS_PER_WORD) * 2;
            data[word_idx] |= (w.to_bits() as u64) << bit_idx;
        }

        Self { data, len }
    }

    /// Quantize an f32 slice using absmean and pack the result.
    ///
    /// Returns `(packed_weights, gamma_scale)`.
    pub fn from_f32_slice(weights: &[f32]) -> (Self, f32) {
        let (ternary, gamma) = absmean_quantize(weights);
        (Self::from_ternary_slice(&ternary), gamma)
    }

    /// Get the ternary weight at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> TernaryWeight {
        debug_assert!(index < self.len, "index {index} out of bounds (len {})", self.len);
        let word_idx = index / WEIGHTS_PER_WORD;
        let bit_idx = (index % WEIGHTS_PER_WORD) * 2;
        let bits = ((self.data[word_idx] >> bit_idx) & 0b11) as u8;
        // Safety: we only store valid 2-bit patterns (0, 1, 2)
        TernaryWeight::from_bits(bits).expect("corrupted packed data")
    }

    /// Set the ternary weight at the given index.
    #[inline]
    pub fn set(&mut self, index: usize, value: TernaryWeight) {
        debug_assert!(index < self.len, "index {index} out of bounds (len {})", self.len);
        let word_idx = index / WEIGHTS_PER_WORD;
        let bit_idx = (index % WEIGHTS_PER_WORD) * 2;
        // Clear the 2-bit field
        self.data[word_idx] &= !(0b11u64 << bit_idx);
        // Set the new value
        self.data[word_idx] |= (value.to_bits() as u64) << bit_idx;
    }

    /// Number of logical weights stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the storage is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Unpack all weights to a Vec of TernaryWeight.
    pub fn to_ternary_vec(&self) -> Vec<TernaryWeight> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    /// Unpack and dequantize to f32 with a scale factor.
    /// Each weight becomes scale * {-1, 0, +1}.
    pub fn to_f32_vec(&self, scale: f32) -> Vec<f32> {
        (0..self.len).map(|i| self.get(i).to_f32() * scale).collect()
    }

    /// Dot product of packed ternary weights with an f32 input vector.
    ///
    /// Computes `scale * sum_i(w_i * x_i)` where `w_i` in {-1, 0, +1}.
    /// This is optimized to avoid actual multiplication: +1 weights add,
    /// -1 weights subtract, 0 weights skip.
    pub fn dot_f32(&self, input: &[f32], scale: f32) -> Result<f32> {
        if input.len() != self.len {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![self.len],
                got: vec![input.len()],
            });
        }

        let mut acc = 0.0f32;
        let full_words = self.len / WEIGHTS_PER_WORD;

        // Process full u64 words
        for word_idx in 0..full_words {
            let word = self.data[word_idx];
            let base = word_idx * WEIGHTS_PER_WORD;
            acc += dot_word(word, &input[base..base + WEIGHTS_PER_WORD]);
        }

        // Process remaining weights in the last partial word
        let remaining = self.len % WEIGHTS_PER_WORD;
        if remaining > 0 {
            let word = self.data[full_words];
            let base = full_words * WEIGHTS_PER_WORD;
            for j in 0..remaining {
                let bits = ((word >> (j * 2)) & 0b11) as u8;
                match bits {
                    1 => acc += input[base + j],
                    2 => acc -= input[base + j],
                    _ => {} // 0 = zero weight, skip
                }
            }
        }

        Ok(acc * scale)
    }

    /// Dot product of a contiguous logical slice against an f32 input vector.
    ///
    /// This is used by packed matrix rows so callers can keep weights packed
    /// during inference instead of materializing a dense dequantized buffer.
    pub fn dot_slice_f32(&self, start: usize, input: &[f32], scale: f32) -> Result<f32> {
        let end = start + input.len();
        if end > self.len {
            return Err(OneBitError::ShapeMismatch {
                expected: vec![self.len.saturating_sub(start)],
                got: vec![input.len()],
            });
        }

        let mut acc = 0.0f32;
        let mut local_offset = 0usize;

        while local_offset < input.len() && ((start + local_offset) % WEIGHTS_PER_WORD) != 0 {
            match self.get(start + local_offset) {
                TernaryWeight::Pos => acc += input[local_offset],
                TernaryWeight::Neg => acc -= input[local_offset],
                TernaryWeight::Zero => {}
            }
            local_offset += 1;
        }

        while local_offset + WEIGHTS_PER_WORD <= input.len() {
            let word_idx = (start + local_offset) / WEIGHTS_PER_WORD;
            let word = self.data[word_idx];
            acc += dot_word(word, &input[local_offset..local_offset + WEIGHTS_PER_WORD]);
            local_offset += WEIGHTS_PER_WORD;
        }

        while local_offset < input.len() {
            match self.get(start + local_offset) {
                TernaryWeight::Pos => acc += input[local_offset],
                TernaryWeight::Neg => acc -= input[local_offset],
                TernaryWeight::Zero => {}
            }
            local_offset += 1;
        }

        Ok(acc * scale)
    }

    /// Number of u64 words in the underlying storage.
    #[inline]
    pub fn word_count(&self) -> usize {
        self.data.len()
    }

    /// Direct access to raw packed storage (for SIMD, serialization).
    #[inline]
    pub fn raw_data(&self) -> &[u64] {
        &self.data
    }

    /// Mutable access to raw packed storage.
    #[inline]
    pub fn raw_data_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Memory usage in bytes (just the packed data, not counting struct overhead).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u64>()
    }
}

/// Process a single u64 word against 32 f32 inputs.
/// Extracts 2-bit ternary encodings and accumulates the dot product.
#[inline]
fn dot_word(word: u64, input: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut w = word;
    for i in 0..WEIGHTS_PER_WORD {
        let bits = (w & 0b11) as u8;
        match bits {
            1 => acc += input[i],
            2 => acc -= input[i],
            _ => {}
        }
        w >>= 2;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
            TernaryWeight::Neg,
        ];
        let packed = PackedTernary::from_ternary_slice(&weights);
        assert_eq!(packed.len(), 5);
        assert_eq!(packed.word_count(), 1); // 5 weights fit in 1 u64

        let unpacked = packed.to_ternary_vec();
        assert_eq!(weights, unpacked);
    }

    #[test]
    fn test_pack_unpack_exact_boundary() {
        // Exactly 32 weights = 1 full word
        let weights: Vec<TernaryWeight> = (0..32)
            .map(|i| match i % 3 {
                0 => TernaryWeight::Pos,
                1 => TernaryWeight::Neg,
                _ => TernaryWeight::Zero,
            })
            .collect();
        let packed = PackedTernary::from_ternary_slice(&weights);
        assert_eq!(packed.word_count(), 1);
        assert_eq!(packed.to_ternary_vec(), weights);
    }

    #[test]
    fn test_pack_unpack_multi_word() {
        // 65 weights = 3 words (64 + 1)
        let weights: Vec<TernaryWeight> = (0..65)
            .map(|i| match i % 3 {
                0 => TernaryWeight::Pos,
                1 => TernaryWeight::Neg,
                _ => TernaryWeight::Zero,
            })
            .collect();
        let packed = PackedTernary::from_ternary_slice(&weights);
        assert_eq!(packed.word_count(), 3); // ceil(65/32) = 3
        assert_eq!(packed.to_ternary_vec(), weights);
    }

    #[test]
    fn test_get_set() {
        let mut packed = PackedTernary::new(10);
        assert_eq!(packed.get(0), TernaryWeight::Zero);

        packed.set(0, TernaryWeight::Pos);
        packed.set(1, TernaryWeight::Neg);
        packed.set(9, TernaryWeight::Pos);

        assert_eq!(packed.get(0), TernaryWeight::Pos);
        assert_eq!(packed.get(1), TernaryWeight::Neg);
        assert_eq!(packed.get(2), TernaryWeight::Zero);
        assert_eq!(packed.get(9), TernaryWeight::Pos);
    }

    #[test]
    fn test_set_overwrites() {
        let mut packed = PackedTernary::new(5);
        packed.set(2, TernaryWeight::Pos);
        assert_eq!(packed.get(2), TernaryWeight::Pos);

        packed.set(2, TernaryWeight::Neg);
        assert_eq!(packed.get(2), TernaryWeight::Neg);

        packed.set(2, TernaryWeight::Zero);
        assert_eq!(packed.get(2), TernaryWeight::Zero);
    }

    #[test]
    fn test_dot_f32_basic() {
        // weights: [+1, -1, 0, +1], input: [1.0, 2.0, 3.0, 4.0]
        // dot = 1*1 + (-1)*2 + 0*3 + 1*4 = 1 - 2 + 0 + 4 = 3.0
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
        ];
        let packed = PackedTernary::from_ternary_slice(&weights);
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = packed.dot_f32(&input, 1.0).unwrap();
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_f32_with_scale() {
        let weights = vec![TernaryWeight::Pos, TernaryWeight::Neg];
        let packed = PackedTernary::from_ternary_slice(&weights);
        let input = vec![1.0, 1.0];

        // dot = scale * (1 - 1) = 0
        let result = packed.dot_f32(&input, 2.5).unwrap();
        assert!((result - 0.0).abs() < 1e-6);

        // weights: [+1, +1], input: [3.0, 4.0], scale = 0.5
        // dot = 0.5 * (3 + 4) = 3.5
        let weights2 = vec![TernaryWeight::Pos, TernaryWeight::Pos];
        let packed2 = PackedTernary::from_ternary_slice(&weights2);
        let result2 = packed2.dot_f32(&[3.0, 4.0], 0.5).unwrap();
        assert!((result2 - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_dot_slice_f32() {
        let weights = vec![
            TernaryWeight::Zero,
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Pos,
            TernaryWeight::Zero,
        ];
        let packed = PackedTernary::from_ternary_slice(&weights);
        let input = vec![2.0f32, 4.0, 8.0];

        let result = packed.dot_slice_f32(1, &input, 0.5).unwrap();
        assert!((result - 0.5 * (2.0 - 4.0 + 8.0)).abs() < 1e-6);
    }

    #[test]
    fn test_dot_f32_large() {
        // Test with >32 weights to exercise multi-word path
        let n = 100;
        let weights: Vec<TernaryWeight> = (0..n)
            .map(|i| match i % 3 {
                0 => TernaryWeight::Pos,
                1 => TernaryWeight::Neg,
                _ => TernaryWeight::Zero,
            })
            .collect();
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let packed = PackedTernary::from_ternary_slice(&weights);

        // Compute expected result manually
        let expected: f32 = weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w.to_f32() * x)
            .sum();

        let result = packed.dot_f32(&input, 1.0).unwrap();
        assert!(
            (result - expected).abs() < 1e-4,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn test_dot_f32_length_mismatch() {
        let packed = PackedTernary::new(5);
        let input = vec![1.0, 2.0, 3.0];
        assert!(packed.dot_f32(&input, 1.0).is_err());
    }

    #[test]
    fn test_from_f32_slice() {
        let weights = vec![1.5, -1.5, 0.01, -0.01, 0.8, -0.8];
        let (packed, gamma) = PackedTernary::from_f32_slice(&weights);
        assert_eq!(packed.len(), 6);
        assert!(gamma > 0.0);

        // All weights should be valid ternary
        for i in 0..6 {
            let w = packed.get(i);
            assert!(
                w == TernaryWeight::Pos
                    || w == TernaryWeight::Neg
                    || w == TernaryWeight::Zero
            );
        }
    }

    #[test]
    fn test_to_f32_vec() {
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Zero,
        ];
        let packed = PackedTernary::from_ternary_slice(&weights);
        let f32_vec = packed.to_f32_vec(2.0);
        assert_eq!(f32_vec, vec![2.0, -2.0, 0.0]);
    }

    #[test]
    fn test_empty() {
        let packed = PackedTernary::new(0);
        assert!(packed.is_empty());
        assert_eq!(packed.len(), 0);
        assert_eq!(packed.word_count(), 0);
    }

    #[test]
    fn test_memory_bytes() {
        let packed = PackedTernary::new(100);
        // 100 weights / 32 per word = 4 words (ceil), 4 * 8 bytes = 32
        assert_eq!(packed.memory_bytes(), 32);
    }
}
