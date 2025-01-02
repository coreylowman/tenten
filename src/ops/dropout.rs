use crate::tensor::*;

impl Tensor {
    pub fn dropout(self, prob: f64) -> Result<Self, Error> {
        let mask = self.sample_uniform_like().le_scalar(prob)?;
        self.div_scalar(1.0 - prob)?.replace(mask, 0.0f64)
    }
}
