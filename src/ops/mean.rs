use crate::tensor::*;

impl Tensor {
    pub fn mean(self) -> Result<Self, Error> {
        let num_elem = self.numel();
        self.sum()?.mul_scalar(1.0f64 / (num_elem as f64))
    }
}
