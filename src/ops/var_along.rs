use crate::{tensor::*};

impl Tensor {
    pub fn var_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();
        self.clone()
            .mean_along(axis)?
            .broadcast_like(axis, &self)?
            .sub(self)?
            .square()?
            .mean_along(axis)
    }
}
