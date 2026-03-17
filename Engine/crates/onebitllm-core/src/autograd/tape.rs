use std::sync::{Arc, Mutex};
use ndarray::{Array, IxDyn};
use super::variable::VarId;

/// A single entry on the computation tape recording one operation.
pub(crate) struct TapeEntry {
    /// ID of the output variable produced by this operation.
    pub output_id: VarId,
    /// IDs of the input variables consumed by this operation.
    pub input_ids: Vec<VarId>,
    /// Backward function: given the gradient of the output, produces gradients
    /// for each input (in the same order as `input_ids`).
    pub backward_fn: Box<dyn FnOnce(&Array<f32, IxDyn>) -> Vec<Array<f32, IxDyn>> + Send>,
}

/// The computation tape that records operations for reverse-mode AD.
///
/// A new tape is created for each forward pass and consumed during backward.
pub struct Tape {
    pub(crate) entries: Vec<TapeEntry>,
    next_id: usize,
}

impl Tape {
    /// Create a new empty tape wrapped in `Arc<Mutex<_>>` for shared ownership.
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            entries: Vec::new(),
            next_id: 0,
        }))
    }

    /// Allocate a fresh unique variable ID.
    pub fn alloc_id(&mut self) -> VarId {
        let id = VarId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Record an operation on the tape.
    ///
    /// - `output_id`: the variable produced by the forward computation.
    /// - `input_ids`: the variables consumed.
    /// - `backward_fn`: closure that, given the output gradient, returns
    ///   a gradient for each input in the same order.
    pub fn record(
        &mut self,
        output_id: VarId,
        input_ids: Vec<VarId>,
        backward_fn: impl FnOnce(&Array<f32, IxDyn>) -> Vec<Array<f32, IxDyn>> + Send + 'static,
    ) {
        self.entries.push(TapeEntry {
            output_id,
            input_ids,
            backward_fn: Box::new(backward_fn),
        });
    }

    /// Return the number of recorded operations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the tape has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Create a tape from existing entries (used by backward pass).
    pub(crate) fn from_entries(entries: Vec<TapeEntry>) -> Self {
        Self {
            entries,
            next_id: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_alloc_ids() {
        let tape = Tape::new();
        let mut t = tape.lock().unwrap();
        let id0 = t.alloc_id();
        let id1 = t.alloc_id();
        let id2 = t.alloc_id();
        assert_eq!(id0, VarId(0));
        assert_eq!(id1, VarId(1));
        assert_eq!(id2, VarId(2));
    }

    #[test]
    fn test_tape_record() {
        let tape = Tape::new();
        let mut t = tape.lock().unwrap();
        let id0 = t.alloc_id();
        let id1 = t.alloc_id();
        let id2 = t.alloc_id();
        t.record(id2, vec![id0, id1], |grad| {
            vec![grad.clone(), grad.clone()]
        });
        assert_eq!(t.len(), 1);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_tape_new_is_empty() {
        let tape = Tape::new();
        let t = tape.lock().unwrap();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }
}
