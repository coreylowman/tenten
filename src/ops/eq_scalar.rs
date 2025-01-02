use crate::tensor::*;

impl Tensor {
    pub fn eq_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let dtype = self.dtype();
        let scalar = Into::<Scalar>::into(scalar).to_dtype(dtype);
        let mut y = self.defer_op_with_args(
            std::format!("eq{}", scalar.to_string()),
            (|x, args| Scalar::Boolean(*x == args[1]), vec![scalar]),
            std::format!("x == {}", scalar.to_string()),
        );
        y.deferred_dtype = Dtype::Boolean;
        Ok(y)
    }
}
