use crate::tensor::*;

impl Tensor {
    pub fn dropout(self, prob: f64) -> Result<Self, Error> {
        let mask = crate::init::sample_uniform_like(&self)?.le_scalar(prob)?;
        self.div_scalar(1.0 - prob)?.replace(mask, 0.0f64)
    }
}
