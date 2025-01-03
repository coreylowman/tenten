use std::{ops::DerefMut, rc::Rc};

use crate::tensor::*;

impl Tensor {
    pub fn fill_with_zeros(&mut self) -> Result<(), Error> {
        match Rc::make_mut(&mut self.bytes).borrow_mut().deref_mut() {
            BytesPtr::Ghost(_, _) => (),
            BytesPtr::Cpu(buf) => buf.fill(0),
            BytesPtr::Cuda(buf) => buf.device().memset_zeros(buf)?,
        }
        self.stored_dtype = self.deferred_dtype;
        self.deferred_ops.clear();
        if let Some(x_grad) = self.grad() {
            crate::backward::record_op(move || x_grad.alloc()?.fill_with_zeros());
        }
        Ok(())
    }
}
