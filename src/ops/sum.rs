use crate::tensor::{Error, Tensor};

impl Tensor {
    pub fn sum(mut self) -> Result<Self, Error> {
        // TODO optimize this into one call. if we can do sum_along in place that'd be great
        for _ in 0..self.shape.len() {
            self = self.sum_along(-1)?;
        }
        assert_eq!(self.shape, vec![]);
        Ok(self)
    }
}
