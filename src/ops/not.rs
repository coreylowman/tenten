use crate::tensor::*;

impl Tensor {
    pub fn not(self) -> Result<Self, Error> {
        assert_eq!(self.dtype(), Dtype::Boolean);
        todo!("defer_op, backwards not supported")
    }
}
