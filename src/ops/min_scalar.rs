use crate::{tensor::*, util::*};

impl Tensor {
    pub fn min_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let dtype = self.dtype();
        let x = self.clone();
        let y = self.defer_op_with_args(
            std::format!("min_{}", scalar.to_string()),
            (
                |a, args| {
                    if *a < args[0] {
                        args[0]
                    } else {
                        *a
                    }
                },
                vec![scalar],
            ),
            std::format!("(x < {v} ? {v} : x)", v = scalar.to_string()),
        );
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.lt_scalar(scalar)?.to_dtype(dtype)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
