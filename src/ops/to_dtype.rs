use crate::{tensor::*, util::*};

impl Tensor {
    pub fn to_dtype(self, dst: Dtype) -> Result<Self, Error> {
        let src = self.dtype();
        if src == dst {
            Ok(self)
        } else if dst.num_bytes() <= self.byte_stride {
            let x = self.clone();
            let mut y: Tensor = self.defer_op_with_args(
                std::format!("to{}", dst.short_name()),
                (|x, args| x.to_dtype(args[0].dtype()), vec![dst.zero()]),
                std::format!("{} x = x", dst.cuda_type_name()),
            );
            y.deferred_dtype = dst;
            if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
                crate::backward::record_op(move || {
                    x_grad.alloc()?.add_assign(y_grad.to_dtype(src)?)
                });
            }
            Ok(y)
        } else {
            todo!(
                "we can still defer this, but ops that do undefer need to reallocate in this case."
            )
        }
    }
}
