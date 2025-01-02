use crate::tensor::*;

impl Tensor {
    pub fn sub_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        self.add_scalar(scalar.negate())
    }
}
