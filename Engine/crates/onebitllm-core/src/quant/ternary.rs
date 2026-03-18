use crate::Result;
use crate::error::OneBitError;

/// A single ternary weight value: {-1, 0, +1}.
///
/// Encoded in 2 bits: 0b00 = Zero, 0b01 = Pos(+1), 0b10 = Neg(-1).
/// The pattern 0b11 is reserved/invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TernaryWeight {
    Zero = 0,
    Pos = 1,
    Neg = 2,
}

impl TernaryWeight {
    /// Encode as 2-bit value.
    #[inline]
    pub fn to_bits(self) -> u8 {
        self as u8
    }

    /// Decode from 2-bit value.
    #[inline]
    pub fn from_bits(bits: u8) -> Result<Self> {
        match bits & 0b11 {
            0 => Ok(TernaryWeight::Zero),
            1 => Ok(TernaryWeight::Pos),
            2 => Ok(TernaryWeight::Neg),
            _ => Err(OneBitError::Quantization(format!(
                "Invalid 2-bit ternary encoding: {bits:#04b}"
            ))),
        }
    }

    /// Convert to f32: Neg -> -1.0, Zero -> 0.0, Pos -> 1.0.
    #[inline]
    pub fn to_f32(self) -> f32 {
        match self {
            TernaryWeight::Neg => -1.0,
            TernaryWeight::Zero => 0.0,
            TernaryWeight::Pos => 1.0,
        }
    }

    /// Quantize a single f32 value given a threshold gamma.
    /// Values > gamma -> Pos, < -gamma -> Neg, else -> Zero.
    #[inline]
    pub fn quantize(value: f32, gamma: f32) -> Self {
        if value > gamma {
            TernaryWeight::Pos
        } else if value < -gamma {
            TernaryWeight::Neg
        } else {
            TernaryWeight::Zero
        }
    }

    /// Quantize to unit ternary using a fixed threshold.
    ///
    /// Values > threshold -> +1, < -threshold -> -1, else -> 0.
    #[inline]
    pub fn quantize_unit(value: f32, threshold: f32) -> Self {
        Self::quantize(value, threshold)
    }
}

impl From<TernaryWeight> for f32 {
    fn from(w: TernaryWeight) -> f32 {
        w.to_f32()
    }
}

impl std::fmt::Display for TernaryWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TernaryWeight::Neg => write!(f, "-1"),
            TernaryWeight::Zero => write!(f, " 0"),
            TernaryWeight::Pos => write!(f, "+1"),
        }
    }
}

/// Quantize an f32 slice to ternary using absmean quantization (BitNet b1.58).
///
/// gamma = mean(|w|) for the slice. Weights above gamma become +1,
/// below -gamma become -1, and the rest become 0.
///
/// Returns `(ternary_weights, gamma_scale)`.
pub fn absmean_quantize(weights: &[f32]) -> (Vec<TernaryWeight>, f32) {
    if weights.is_empty() {
        return (Vec::new(), 0.0);
    }

    let gamma: f32 = weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len() as f32;

    let ternary: Vec<TernaryWeight> = weights
        .iter()
        .map(|&w| TernaryWeight::quantize(w, gamma))
        .collect();

    (ternary, gamma)
}

/// Round-clip quantization: quantize using a provided gamma threshold.
pub fn round_clip_quantize(weights: &[f32], gamma: f32) -> Vec<TernaryWeight> {
    weights
        .iter()
        .map(|&w| TernaryWeight::quantize(w, gamma))
        .collect()
}

/// Quantize an f32 slice to unit ternary {-1, 0, +1} with a fixed threshold.
pub fn unit_quantize(weights: &[f32], threshold: f32) -> Vec<TernaryWeight> {
    weights
        .iter()
        .map(|&w| TernaryWeight::quantize_unit(w, threshold))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_bits_roundtrip() {
        for w in [TernaryWeight::Neg, TernaryWeight::Zero, TernaryWeight::Pos] {
            let bits = w.to_bits();
            let decoded = TernaryWeight::from_bits(bits).unwrap();
            assert_eq!(w, decoded);
        }
    }

    #[test]
    fn test_ternary_invalid_bits() {
        assert!(TernaryWeight::from_bits(3).is_err());
    }

    #[test]
    fn test_ternary_to_f32() {
        assert_eq!(TernaryWeight::Neg.to_f32(), -1.0);
        assert_eq!(TernaryWeight::Zero.to_f32(), 0.0);
        assert_eq!(TernaryWeight::Pos.to_f32(), 1.0);
    }

    #[test]
    fn test_quantize_threshold() {
        assert_eq!(TernaryWeight::quantize(0.5, 0.3), TernaryWeight::Pos);
        assert_eq!(TernaryWeight::quantize(-0.5, 0.3), TernaryWeight::Neg);
        assert_eq!(TernaryWeight::quantize(0.1, 0.3), TernaryWeight::Zero);
        assert_eq!(TernaryWeight::quantize(0.0, 0.3), TernaryWeight::Zero);
    }

    #[test]
    fn test_absmean_quantize() {
        let weights = vec![1.0, -1.0, 0.1, -0.1, 0.5, -0.5];
        let (ternary, gamma) = absmean_quantize(&weights);

        // gamma = mean(|w|) = (1 + 1 + 0.1 + 0.1 + 0.5 + 0.5) / 6 ≈ 0.533
        assert!(gamma > 0.5 && gamma < 0.6);

        assert_eq!(ternary[0], TernaryWeight::Pos); // 1.0 > 0.533
        assert_eq!(ternary[1], TernaryWeight::Neg); // -1.0 < -0.533
        assert_eq!(ternary[2], TernaryWeight::Zero); // 0.1 in [-0.533, 0.533]
        assert_eq!(ternary[3], TernaryWeight::Zero); // -0.1
        assert_eq!(ternary[4], TernaryWeight::Zero); // 0.5 < 0.533
        assert_eq!(ternary[5], TernaryWeight::Zero); // -0.5 > -0.533
    }

    #[test]
    fn test_absmean_empty() {
        let (t, g) = absmean_quantize(&[]);
        assert!(t.is_empty());
        assert_eq!(g, 0.0);
    }

    #[test]
    fn test_round_clip() {
        let weights = vec![2.0, -2.0, 0.0];
        let ternary = round_clip_quantize(&weights, 0.5);
        assert_eq!(ternary[0], TernaryWeight::Pos);
        assert_eq!(ternary[1], TernaryWeight::Neg);
        assert_eq!(ternary[2], TernaryWeight::Zero);
    }

    #[test]
    fn test_unit_quantize() {
        let weights = vec![0.8, -0.8, 0.1, -0.1];
        let ternary = unit_quantize(&weights, 0.5);
        assert_eq!(ternary[0], TernaryWeight::Pos);
        assert_eq!(ternary[1], TernaryWeight::Neg);
        assert_eq!(ternary[2], TernaryWeight::Zero);
        assert_eq!(ternary[3], TernaryWeight::Zero);
    }
}
