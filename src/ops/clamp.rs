use crate::{tensor::*};

impl Tensor {
    pub fn clamp<S: Into<Scalar>>(self, min: S, max: S) -> Result<Self, Error> {
        self.max_scalar(min)?.min_scalar(max)
    }
}
