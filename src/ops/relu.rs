use crate::tensor::*;

impl Tensor {
    pub fn relu(self) -> Result<Self, Error> {
        let zero = self.dtype().zero();
        self.max_scalar(zero)
    }
}
