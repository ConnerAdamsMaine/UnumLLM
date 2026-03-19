use crate::error::OneBitError;
use crate::Result;

/// Number of binary weights packed per u64 word (1 bit each).
const WEIGHTS_PER_WORD: usize = 64;

/// Bitpacked binary weight storage.
///
/// Each weight occupies a single bit within u64 words. Bit `i` is stored in
/// bit position `i % 64` of word `i / 64`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBinary {
    data: Vec<u64>,
    len: usize,
}

impl PackedBinary {
    /// Create a new packed storage initialized to zero bits.
    pub fn new(len: usize) -> Self {
        let num_words = len.div_ceil(WEIGHTS_PER_WORD);
        Self {
            data: vec![0u64; num_words],
            len,
        }
    }

    /// Pack a slice of boolean bits.
    pub fn from_bool_slice(bits: &[bool]) -> Self {
        let mut packed = Self::new(bits.len());
        for (index, &bit) in bits.iter().enumerate() {
            packed.set(index, bit);
        }
        packed
    }

    /// Rebuild packed storage from raw u64 words and the logical bit count.
    pub fn from_raw_parts(data: Vec<u64>, len: usize) -> Result<Self> {
        let expected_words = len.div_ceil(WEIGHTS_PER_WORD);
        if data.len() != expected_words {
            return Err(OneBitError::Config(format!(
                "packed binary word count mismatch: expected {expected_words}, got {}",
                data.len()
            )));
        }
        Ok(Self { data, len })
    }

    /// Get the stored bit at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        debug_assert!(
            index < self.len,
            "index {index} out of bounds (len {})",
            self.len
        );
        let word_idx = index / WEIGHTS_PER_WORD;
        let bit_idx = index % WEIGHTS_PER_WORD;
        ((self.data[word_idx] >> bit_idx) & 1) != 0
    }

    /// Set the stored bit at the given index.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        debug_assert!(
            index < self.len,
            "index {index} out of bounds (len {})",
            self.len
        );
        let word_idx = index / WEIGHTS_PER_WORD;
        let bit_idx = index % WEIGHTS_PER_WORD;
        self.data[word_idx] &= !(1u64 << bit_idx);
        self.data[word_idx] |= (value as u64) << bit_idx;
    }

    /// Number of logical bits stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the storage is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Unpack all stored bits to a Vec of bools.
    pub fn to_bool_vec(&self) -> Vec<bool> {
        (0..self.len).map(|index| self.get(index)).collect()
    }

    /// Number of u64 words in the underlying storage.
    #[inline]
    pub fn word_count(&self) -> usize {
        self.data.len()
    }

    /// Direct access to raw packed storage (for serialization).
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

/// Deterministic metadata-driven base sign for a packed binary weight index.
///
/// `false` represents a negative base sign and `true` represents a positive
/// base sign. This is intentionally cheap and stable across platforms.
pub fn equalizer_base_sign(seed: u64, index: usize) -> bool {
    (splitmix64(seed ^ index as u64) & 1) != 0
}

/// Convert a stored binary toggle bit back into an effective signed weight.
///
/// The raw bitstream stores XOR toggles relative to the metadata-defined base
/// sign. This reconstructs the logical weight in `{-1.0, +1.0}`.
pub fn effective_sign_from_toggle(stored_bit: bool, seed: u64, index: usize) -> f32 {
    if stored_bit ^ equalizer_base_sign(seed, index) {
        1.0
    } else {
        -1.0
    }
}

/// Encode an effective signed weight into the stored binary toggle bit.
pub fn toggle_bit_for_sign(is_positive: bool, seed: u64, index: usize) -> bool {
    is_positive ^ equalizer_base_sign(seed, index)
}

#[inline]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let bits = vec![true, false, true, true, false, false, true];
        let packed = PackedBinary::from_bool_slice(&bits);
        assert_eq!(packed.to_bool_vec(), bits);
    }

    #[test]
    fn test_from_raw_parts_rejects_wrong_word_count() {
        let err = PackedBinary::from_raw_parts(vec![], 65).unwrap_err();
        assert!(err
            .to_string()
            .contains("packed binary word count mismatch"));
    }

    #[test]
    fn test_equalizer_toggle_roundtrip() {
        let seed = 0x1B17_EA11_u64;
        for (index, is_positive) in [true, false, true, false].into_iter().enumerate() {
            let stored = toggle_bit_for_sign(is_positive, seed, index);
            let rebuilt = effective_sign_from_toggle(stored, seed, index);
            assert_eq!(rebuilt > 0.0, is_positive);
        }
    }
}
