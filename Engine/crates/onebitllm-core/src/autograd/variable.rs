use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use ndarray::{Array, IxDyn};
use super::tape::Tape;
use super::backward::run_backward;

/// A unique identifier for a variable on the computation tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarId(pub(crate) usize);

/// A tracked tensor value that can participate in automatic differentiation.
///
/// Variables hold their data, a reference to the computation tape,
/// and whether they require gradient computation.
#[derive(Clone)]
pub struct Variable {
    /// Unique identifier on the tape.
    pub id: VarId,
    /// The tensor data (f32, dynamic dimensionality).
    pub data: Array<f32, IxDyn>,
    /// Whether this variable requires gradient computation.
    pub requires_grad: bool,
    /// Reference to the computation tape (None for detached variables).
    pub(crate) tape: Option<Arc<Mutex<Tape>>>,
}

impl Variable {
    /// Create a new variable tracked on the given tape.
    pub fn new(data: Array<f32, IxDyn>, requires_grad: bool, tape: &Arc<Mutex<Tape>>) -> Self {
        let id = tape.lock().unwrap().alloc_id();
        Self {
            id,
            data,
            requires_grad,
            tape: Some(Arc::clone(tape)),
        }
    }

    /// Create a detached variable (not on any tape, no gradient tracking).
    pub fn detached(data: Array<f32, IxDyn>) -> Self {
        Self {
            id: VarId(usize::MAX),
            data,
            requires_grad: false,
            tape: None,
        }
    }

    /// Run the backward pass from this variable, returning gradients for all
    /// upstream variables that require them.
    ///
    /// The tape is consumed during backward and cannot be reused.
    /// The seed gradient is `ones_like(self.data)` for scalar outputs,
    /// or must be provided explicitly for non-scalar outputs.
    pub fn backward(&self) -> crate::Result<HashMap<VarId, Array<f32, IxDyn>>> {
        let tape_ref = self.tape.as_ref().ok_or_else(|| {
            crate::error::OneBitError::Autograd(
                "Cannot call backward on a detached variable".into(),
            )
        })?;

        // Take the tape entries out, consuming the tape state.
        let tape = {
            let mut t = tape_ref.lock().unwrap();
            let entries = std::mem::take(&mut t.entries);
            super::tape::Tape::from_entries(entries)
        };

        // Seed gradient: ones with the same shape as the output.
        let seed = Array::ones(self.data.raw_dim());

        Ok(run_backward(tape, self.id, seed))
    }
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Variable")
            .field("id", &self.id)
            .field("shape", &self.data.shape())
            .field("requires_grad", &self.requires_grad)
            .field("on_tape", &self.tape.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_new() {
        let tape = Tape::new();
        let data = Array::from_elem(IxDyn(&[2, 3]), 1.0f32);
        let v = Variable::new(data.clone(), true, &tape);
        assert_eq!(v.id, VarId(0));
        assert!(v.requires_grad);
        assert!(v.tape.is_some());
        assert_eq!(v.data.shape(), &[2, 3]);
    }

    #[test]
    fn test_variable_detached() {
        let data = Array::from_elem(IxDyn(&[4]), 2.0f32);
        let v = Variable::detached(data);
        assert!(!v.requires_grad);
        assert!(v.tape.is_none());
    }

    #[test]
    fn test_backward_on_detached_fails() {
        let v = Variable::detached(Array::from_elem(IxDyn(&[1]), 1.0f32));
        assert!(v.backward().is_err());
    }

    #[test]
    fn test_multiple_vars_on_tape() {
        let tape = Tape::new();
        let v0 = Variable::new(Array::from_elem(IxDyn(&[1]), 1.0f32), true, &tape);
        let v1 = Variable::new(Array::from_elem(IxDyn(&[1]), 2.0f32), true, &tape);
        let v2 = Variable::new(Array::from_elem(IxDyn(&[1]), 3.0f32), false, &tape);
        assert_eq!(v0.id, VarId(0));
        assert_eq!(v1.id, VarId(1));
        assert_eq!(v2.id, VarId(2));
    }
}
