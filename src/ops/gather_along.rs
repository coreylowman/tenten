use crate::tensor::*;

impl Tensor {
    pub fn gather_along<A: Into<Axis>>(self, axis: A, indices: Self) -> Result<Self, Error> {
        todo!()
    }
}
