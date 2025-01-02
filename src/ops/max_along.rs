use crate::tensor::*;

impl Tensor {
    pub fn max_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        todo!()
    }
}
