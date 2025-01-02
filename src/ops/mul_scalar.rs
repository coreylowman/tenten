use crate::{tensor::*, util::all_some};

impl Tensor {
    pub fn mul_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let x = self.clone();
        let y = self.defer_op_with_args(
            std::format!("mul{}", scalar.to_string()),
            (|x, args| *x * args[0], vec![scalar]),
            std::format!("x * {}", scalar.to_string()),
        );
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                x_grad.alloc()?.add_assign(y_grad.mul_scalar(scalar)?)
            });
        }
        Ok(y)
    }
}
