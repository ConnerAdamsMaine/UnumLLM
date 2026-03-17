/// SIMD-optimized kernels for ternary operations.
///
/// Currently provides scalar fallback implementations. When the `simd` feature
/// is enabled, platform-specific SIMD paths can be added.

/// Ternary dot product: scalar fallback.
///
/// Processes packed u64 words containing 2-bit ternary weights and computes
/// the dot product with an f32 input vector.
pub fn scalar_ternary_dot(packed: &[u64], input: &[f32], len: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut input_idx = 0;

    for &word in packed {
        let mut w = word;
        let remaining = len - input_idx;
        let count = remaining.min(32);

        for _ in 0..count {
            let bits = (w & 0b11) as u8;
            match bits {
                1 => acc += input[input_idx],
                2 => acc -= input[input_idx],
                _ => {}
            }
            w >>= 2;
            input_idx += 1;
        }

        if input_idx >= len {
            break;
        }
    }

    acc
}

/// SIMD-accelerated ternary dot product (placeholder).
///
/// When the `simd` feature is enabled, this will use platform intrinsics.
/// Currently dispatches to the scalar implementation.
#[cfg(feature = "simd")]
pub fn simd_ternary_dot(packed: &[u64], input: &[f32], len: usize) -> f32 {
    // TODO: implement with std::arch intrinsics for x86_64 / aarch64
    scalar_ternary_dot(packed, input, len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::TernaryWeight;

    #[test]
    fn test_scalar_dot_matches_bitpack() {
        let weights = vec![
            TernaryWeight::Pos,
            TernaryWeight::Neg,
            TernaryWeight::Zero,
            TernaryWeight::Pos,
            TernaryWeight::Neg,
        ];
        let packed = crate::quant::PackedTernary::from_ternary_slice(&weights);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let expected = packed.dot_f32(&input, 1.0).unwrap();
        let result = scalar_ternary_dot(packed.raw_data(), &input, packed.len());
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_dot_large() {
        let n = 100;
        let weights: Vec<TernaryWeight> = (0..n)
            .map(|i| match i % 3 {
                0 => TernaryWeight::Pos,
                1 => TernaryWeight::Neg,
                _ => TernaryWeight::Zero,
            })
            .collect();
        let packed = crate::quant::PackedTernary::from_ternary_slice(&weights);
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();

        let expected: f32 = weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w.to_f32() * x)
            .sum();

        let result = scalar_ternary_dot(packed.raw_data(), &input, n);
        assert!((result - expected).abs() < 1e-4);
    }
}
