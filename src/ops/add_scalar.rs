use crate::{tensor::*, util::*};

impl Tensor {
    pub fn add_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let x_ghost = self.clone();
        let y = self.defer_op_with_args(
            std::format!("add{}", scalar.to_string()),
            (|x, args| *x + args[0], vec![scalar]),
            std::format!("$x + {scalar:?}"),
        );
        if let Some([x_grad, y_grad]) = all_some([x_ghost.grad(), y.grad()]) {
            crate::backward::record_op(move || x_grad.alloc()?.add_assign(y_grad));
        }
        Ok(y)
    }
}
