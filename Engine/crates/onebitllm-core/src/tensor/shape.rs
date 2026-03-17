use crate::Result;
use crate::error::OneBitError;

/// Compute the flat index from multi-dimensional indices (row-major/C-order).
pub fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    debug_assert_eq!(indices.len(), shape.len());
    let mut flat = 0;
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        flat += indices[i] * stride;
        stride *= shape[i];
    }
    flat
}

/// Compute multi-dimensional indices from flat index (row-major/C-order).
pub fn unravel_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = flat;
    for i in (0..shape.len()).rev() {
        indices[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    indices
}

/// Compute strides for a given shape (row-major/C-order).
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Check if two shapes are broadcast-compatible, returning the resulting shape.
///
/// Follows NumPy broadcasting rules: dimensions are compared from trailing,
/// and dimensions are compatible when they are equal or one of them is 1.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = vec![0; max_ndim];

    for i in 0..max_ndim {
        let da = if i < a.len() {
            a[a.len() - 1 - i]
        } else {
            1
        };
        let db = if i < b.len() {
            b[b.len() - 1 - i]
        } else {
            1
        };

        if da == db {
            result[max_ndim - 1 - i] = da;
        } else if da == 1 {
            result[max_ndim - 1 - i] = db;
        } else if db == 1 {
            result[max_ndim - 1 - i] = da;
        } else {
            return Err(OneBitError::ShapeMismatch {
                expected: a.to_vec(),
                got: b.to_vec(),
            });
        }
    }

    Ok(result)
}

/// Validate that a reshape is valid (total number of elements unchanged).
pub fn validate_reshape(old_shape: &[usize], new_shape: &[usize]) -> Result<()> {
    let old_total: usize = old_shape.iter().product();
    let new_total: usize = new_shape.iter().product();
    if old_total != new_total {
        return Err(OneBitError::TensorOp(format!(
            "Cannot reshape from {:?} (total {old_total}) to {:?} (total {new_total})",
            old_shape, new_shape
        )));
    }
    Ok(())
}

/// Total number of elements for a given shape.
pub fn num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ravel_unravel_roundtrip() {
        let shape = [3, 4, 5];
        for flat in 0..60 {
            let indices = unravel_index(flat, &shape);
            let back = ravel_index(&indices, &shape);
            assert_eq!(flat, back);
        }
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[3, 4, 5]), vec![20, 5, 1]);
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_broadcast_shape_same() {
        assert_eq!(broadcast_shape(&[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shape_scalar() {
        assert_eq!(broadcast_shape(&[3, 4], &[1]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shape(&[1], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shape_expand() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
        assert_eq!(
            broadcast_shape(&[1, 3, 1], &[2, 1, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn test_broadcast_shape_different_ndim() {
        assert_eq!(broadcast_shape(&[3, 4], &[4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shape_incompatible() {
        assert!(broadcast_shape(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn test_validate_reshape() {
        assert!(validate_reshape(&[2, 3], &[6]).is_ok());
        assert!(validate_reshape(&[2, 3], &[3, 2]).is_ok());
        assert!(validate_reshape(&[2, 3], &[5]).is_err());
    }
}
