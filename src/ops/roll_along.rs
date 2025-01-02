use crate::tensor::*;

impl Tensor {
    pub fn roll_along<A: Into<Axis>>(self, axis: A, shift: isize) -> Result<Self, Error> {
        todo!()
    }
}
