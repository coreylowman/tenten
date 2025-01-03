use crate::tensor::*;

impl Tensor {
    pub fn ne_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let mut y = self.defer_op_with_args(
            std::format!("ne_{}", scalar.to_string()),
            (|x, args| Scalar::Boolean(*x != args[1]), vec![scalar]),
            std::format!("bool $x1 = $x != {}", scalar.to_string()),
        );
        y.deferred_dtype = Dtype::Boolean;
        y.gradient = None;
        Ok(y)
    }
}
