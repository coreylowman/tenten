use crate::tensor::*;

impl Tensor {
    pub fn or(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            return Ok(self);
        }

        assert_eq!(self.dtype(), Dtype::Boolean);
        assert_eq!(other.dtype(), Dtype::Boolean);
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("defer_op, backwards not supported")
    }
}
