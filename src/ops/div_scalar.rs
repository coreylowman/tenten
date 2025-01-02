use crate::tensor::*;

impl Tensor {
    pub fn div_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        match self.dtype() {
            Dtype::Float16 | Dtype::BFloat16 | Dtype::Float32 | Dtype::Float64 => {
                self.mul_scalar(scalar.recip())
            }
            _ => todo!(),
        }
    }
}
