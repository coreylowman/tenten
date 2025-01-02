use crate::{init::nd_bytes_strides, tensor::*, util::all_some};

impl Tensor {
    pub fn reshape_like(self, other: &Self) -> Result<Self, Error> {
        self.reshape(other.shape())
    }

    pub fn contiguous(self) -> Result<Self, Error> {
        let shape = self.shape.clone();
        self.reshape(shape)
    }

    pub fn reshape<Shape: Into<Vec<usize>>>(mut self, shape: Shape) -> Result<Self, Error> {
        let shape = Into::<Vec<usize>>::into(shape);
        let old_numel = self.numel();
        let new_numel: usize = shape.iter().product();
        assert_eq!(old_numel, new_numel, "Can't reshape tensor with {old_numel} elements into a shape ({shape:?}) with {new_numel} elements.");
        if self.strides == nd_bytes_strides(&shape, self.byte_stride) {
            // no-op
            Ok(self)
        } else if self.strides == nd_bytes_strides(&self.shape, self.byte_stride) {
            // already contiguous, just change shape & strides
            self.strides = nd_bytes_strides(&shape, self.byte_stride);
            self.shape = shape;
            if let Some([x_grad, y_grad]) = all_some([self.grad(), self.set_new_grad()]) {
                crate::backward::record_op(move || {
                    let y_grad = y_grad.reshape_like(&x_grad)?;
                    x_grad.alloc()?.add_assign(y_grad)
                });
            }
            Ok(self)
        } else {
            // need to make contiguous & update shape/strides
            todo!()
        }
    }
}
