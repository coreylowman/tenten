use crate::tensor::*;

impl Tensor {
    pub fn replace<S: Into<Scalar>>(self, cond: Self, value: S) -> Result<Self, Error> {
        assert_eq!(cond.dtype(), Dtype::Boolean);
        assert_eq!(self.shape(), cond.shape());
        assert_eq!(self.device(), cond.device());

        let scalar = Into::<Scalar>::into(value).to_dtype(self.dtype());
        todo!()
    }
}
