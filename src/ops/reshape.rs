use crate::tensor::*;

impl Tensor {
    pub fn reshape_like(self, other: &Self) -> Result<Self, Error> {
        self.reshape(other.shape())
    }

    pub fn contiguous(self) -> Result<Self, Error> {
        let shape = self.shape.clone();
        self.reshape(shape)
    }

    pub fn reshape<Shape: Into<Vec<usize>>>(self, shape: Shape) -> Result<Self, Error> {
        todo!()
    }
}
