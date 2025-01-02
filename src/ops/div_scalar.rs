use crate::tensor::*;

impl Tensor {
    pub fn div_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        // TODO handle integer values
        self.mul_scalar(scalar.recip())
    }
}
