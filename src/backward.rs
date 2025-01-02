use std::cell::RefCell;

use crate::tensor::*;

pub(crate) struct Tape {
    backward_ops: Vec<(
        MonotonicallyIncreasingId,
        Box<dyn FnOnce() -> Result<(), Error>>,
    )>,
}

#[derive(Debug, Clone)]
pub struct MonotonicallyIncreasingId(pub(crate) u64);

#[inline(always)]
pub fn monotonically_increasing_id() -> MonotonicallyIncreasingId {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    MonotonicallyIncreasingId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
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
            tape.backward_ops
                .push((monotonically_increasing_id(), Box::new(f)));
        })
    })
}

pub fn is_recording() -> bool {
    IS_RECORDING.with_borrow(|is_recording| *is_recording)
}

pub fn set_recording(record_grads: bool) {
    IS_RECORDING.with_borrow_mut(|is_recording| *is_recording = record_grads)
}

pub struct WithGradGuard {
    prev: bool,
}
pub fn with_grad() -> WithGradGuard {
    WithGradGuard {
        prev: IS_RECORDING.with_borrow_mut(|x| std::mem::replace(x, true)),
    }
}
impl Drop for WithGradGuard {
    fn drop(&mut self) {
        IS_RECORDING.with_borrow_mut(|x| *x = self.prev);
    }
}

pub struct NoGradGuard {
    prev: bool,
}
pub fn no_grad() -> NoGradGuard {
    NoGradGuard {
        prev: IS_RECORDING.with_borrow_mut(|x| std::mem::replace(x, false)),
    }
}
impl Drop for NoGradGuard {
    fn drop(&mut self) {
        IS_RECORDING.with_borrow_mut(|x| *x = self.prev);
    }
}

impl Tensor {
    pub fn backward(self) -> Result<(), Error> {
        assert!(self.shape().len() == 0);
        let grad = self.grad().expect("Loss didn't have gradient");
        grad.alloc()?.fill_with(self.dtype().one())?;
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
