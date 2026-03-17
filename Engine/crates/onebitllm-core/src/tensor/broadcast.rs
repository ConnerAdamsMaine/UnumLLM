use ndarray::{Array, IxDyn};

use crate::Result;
use crate::error::OneBitError;

/// Broadcast a dense f32 array to a target shape.
///
/// Uses ndarray's built-in broadcasting. If the array cannot be broadcast
/// to the target shape, returns an error.
pub fn broadcast_to(arr: &Array<f32, IxDyn>, target_shape: &[usize]) -> Result<Array<f32, IxDyn>> {
    let target = IxDyn(target_shape);
    arr.broadcast(target)
        .map(|view| view.to_owned())
        .ok_or_else(|| OneBitError::ShapeMismatch {
            expected: target_shape.to_vec(),
            got: arr.shape().to_vec(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_broadcast_scalar() {
        let arr = Array::from_elem(IxDyn(&[1]), 5.0f32);
        let result = broadcast_to(&arr, &[3]).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result, Array::from_elem(IxDyn(&[3]), 5.0f32));
    }

    #[test]
    fn test_broadcast_row_to_matrix() {
        let arr = array![1.0f32, 2.0, 3.0].into_dyn();
        let result = broadcast_to(&arr, &[2, 3]).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 2]], 3.0);
    }

    #[test]
    fn test_broadcast_incompatible() {
        let arr = Array::from_elem(IxDyn(&[3]), 1.0f32);
        assert!(broadcast_to(&arr, &[2]).is_err());
    }
}
