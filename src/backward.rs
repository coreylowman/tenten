use std::cell::RefCell;

use crate::tensor::*;

pub(crate) struct Tape {
    backward_ops: Vec<(UniqueId, Box<dyn FnOnce() -> Result<(), Error>>)>,
}

thread_local! {
    pub(crate) static TAPE: RefCell<Tape> = const {
        RefCell::new(Tape {
            backward_ops: Vec::new()
        })
    }
}

thread_local! {
    /// We need this separately from TAPE because during backwards we will but mutating TAPE in a loop
    /// while also executing backward ops on tensor.
    pub(crate) static IS_RECORDING: RefCell<bool> = const { RefCell::new(true) }
}

#[inline]
pub fn record_op<F: 'static + FnOnce() -> Result<(), Error>>(f: F) {
    IS_RECORDING.with_borrow(|&is_recording| {
        assert!(is_recording);
        TAPE.with_borrow_mut(|tape| {
            tape.backward_ops.push((unique_id(), Box::new(f)));
        })
    })
}

pub fn is_recording() -> bool {
    IS_RECORDING.with_borrow(|is_recording| *is_recording)
}

pub struct WithGradGuard {
    was_recording: bool,
}
pub fn with_grad() -> WithGradGuard {
    WithGradGuard {
        was_recording: IS_RECORDING
            .with_borrow_mut(|is_recording| std::mem::replace(is_recording, true)),
    }
}
impl Drop for WithGradGuard {
    fn drop(&mut self) {
        IS_RECORDING.with_borrow_mut(|is_recording| *is_recording = self.was_recording);
    }
}

pub struct NoGradGuard {
    was_recording: bool,
}
pub fn no_grad() -> NoGradGuard {
    NoGradGuard {
        was_recording: IS_RECORDING
            .with_borrow_mut(|is_recording| std::mem::replace(is_recording, false)),
    }
}
impl Drop for NoGradGuard {
    fn drop(&mut self) {
        IS_RECORDING.with_borrow_mut(|is_recording| *is_recording = self.was_recording);
    }
}

impl Tensor {
    pub fn backward(self) -> Result<(), Error> {
        assert!(self.shape().len() == 0);
        let loss = self.undefer()?;
        let grad = loss.grad().expect("Loss didn't have gradient");
        grad.alloc()?.fill_with_ones()?;
        TAPE.with_borrow_mut(|tape| {
            let _no_grad = no_grad();
            tape.backward_ops.sort_by_key(|k| k.0 .0);
            for (_uid, backward_op) in tape.backward_ops.drain(..).rev() {
                backward_op()?;
            }
            Ok(())
        })
    }
}
