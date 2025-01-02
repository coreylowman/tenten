use crate::tensor::*;

impl Tensor {
    pub fn chunk_along<A: Into<Axis>>(self, num_chunks: usize, axis: A) -> Result<Self, Error> {
        todo!()
    }
}
