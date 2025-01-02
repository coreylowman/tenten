use crate::tensor::*;

impl Tensor {
    pub fn slice_along<A: Into<Axis>, R: std::ops::RangeBounds<usize>>(
        self,
        axis: A,
        range: R,
    ) -> Result<Self, Error> {
        todo!()
    }
}
