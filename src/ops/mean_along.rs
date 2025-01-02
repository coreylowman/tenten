use crate::{tensor::*};

impl Tensor {
    pub fn mean_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let dim = axis.get_value(&self.shape);
        self.sum_along(axis)?.mul_scalar(1.0f64 / (dim as f64))
    }
}
