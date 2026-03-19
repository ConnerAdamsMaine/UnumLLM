use super::tape::Tape;
use super::variable::VarId;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;

/// Run the backward pass by traversing tape entries in reverse order.
///
/// Starting from the output variable with the given seed gradient,
/// propagates gradients backward through the recorded operations.
/// Returns a map from VarId to accumulated gradient.
pub fn run_backward(
    tape: Tape,
    output_id: VarId,
    seed_grad: Array<f32, IxDyn>,
) -> HashMap<VarId, Array<f32, IxDyn>> {
    let mut grads: HashMap<VarId, Array<f32, IxDyn>> = HashMap::new();
    grads.insert(output_id, seed_grad);

    // Traverse entries in reverse (topological order from output to inputs).
    for entry in tape.entries.into_iter().rev() {
        // Get the gradient for this entry's output; skip if none accumulated.
        let out_grad = match grads.get(&entry.output_id) {
            Some(g) => g.clone(),
            None => continue,
        };

        // Call the backward function to get input gradients.
        let input_grads = (entry.backward_fn)(&out_grad);

        // Accumulate gradients for each input.
        debug_assert_eq!(
            input_grads.len(),
            entry.input_ids.len(),
            "backward_fn returned {} gradients but expected {}",
            input_grads.len(),
            entry.input_ids.len()
        );

        for (id, grad) in entry.input_ids.into_iter().zip(input_grads.into_iter()) {
            grads
                .entry(id)
                .and_modify(|existing| *existing = &*existing + &grad)
                .or_insert(grad);
        }
    }

    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::tape::Tape as TapeStruct;

    #[test]
    fn test_backward_linear_chain() {
        // Build a simple chain: x -> y = 2*x -> z = y + 1
        // dz/dx should be 2.0

        let tape_arc = TapeStruct::new();
        let (x_id, y_id, z_id);
        {
            let mut tape = tape_arc.lock().unwrap();
            x_id = tape.alloc_id();
            y_id = tape.alloc_id();
            z_id = tape.alloc_id();

            // y = 2 * x: backward: grad_x = 2 * grad_y
            tape.record(y_id, vec![x_id], |grad| vec![grad * 2.0]);

            // z = y + 1: backward: grad_y = grad_z
            tape.record(z_id, vec![y_id], |grad| vec![grad.clone()]);
        }

        // Extract tape for backward
        let tape = {
            let mut t = tape_arc.lock().unwrap();
            let entries = std::mem::take(&mut t.entries);
            TapeStruct::from_entries(entries)
        };

        let seed = Array::from_elem(IxDyn(&[1]), 1.0f32);
        let grads = run_backward(tape, z_id, seed);

        let grad_x = grads.get(&x_id).unwrap();
        assert_eq!(grad_x.as_slice().unwrap(), &[2.0]);
    }

    #[test]
    fn test_backward_fan_out() {
        // x -> y1 = x, y2 = x -> z = y1 + y2
        // dz/dx = 2.0 (gradient accumulation)

        let tape_arc = TapeStruct::new();
        let (x_id, y1_id, y2_id, z_id);
        {
            let mut tape = tape_arc.lock().unwrap();
            x_id = tape.alloc_id();
            y1_id = tape.alloc_id();
            y2_id = tape.alloc_id();
            z_id = tape.alloc_id();

            // y1 = x (identity)
            tape.record(y1_id, vec![x_id], |grad| vec![grad.clone()]);

            // y2 = x (identity)
            tape.record(y2_id, vec![x_id], |grad| vec![grad.clone()]);

            // z = y1 + y2
            tape.record(z_id, vec![y1_id, y2_id], |grad| {
                vec![grad.clone(), grad.clone()]
            });
        }

        let tape = {
            let mut t = tape_arc.lock().unwrap();
            let entries = std::mem::take(&mut t.entries);
            TapeStruct::from_entries(entries)
        };

        let seed = Array::from_elem(IxDyn(&[1]), 1.0f32);
        let grads = run_backward(tape, z_id, seed);

        let grad_x = grads.get(&x_id).unwrap();
        assert_eq!(grad_x.as_slice().unwrap(), &[2.0]);
    }
}
