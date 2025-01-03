use std::{ops::DerefMut, rc::Rc};

use cudarc::driver::DeviceSlice;

use crate::tensor::*;

impl Tensor {
    pub fn fill_with<S>(&mut self, scalar: S) -> Result<(), Error>
    where
        S: Into<Scalar>,
    {
        let dtype = self.deferred_dtype;
        let value = Into::<Scalar>::into(scalar).to_dtype(dtype);

        match Rc::make_mut(&mut self.bytes).borrow_mut().deref_mut() {
            BytesPtr::Ghost(_, _) => (),
            BytesPtr::Cpu(buf) => {
                for i in (0..buf.len()).step_by(dtype.num_bytes()) {
                    value.store(&mut buf[i..]);
                }
            }
            BytesPtr::Cuda(buf) => {
                let mut init_buf = vec![0u8; buf.len()];
                for i in (0..init_buf.len()).step_by(dtype.num_bytes()) {
                    value.store(&mut init_buf[i..]);
                }
                buf.device().htod_copy_into(init_buf, buf)?
            }
        };
        self.stored_dtype = dtype;
        self.deferred_ops.clear();
        if let Some(x_grad) = self.grad() {
            crate::backward::record_op(move || x_grad.alloc()?.fill_with_zeros());
        }
        Ok(())
    }
}
