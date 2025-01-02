use crate::{tensor::*};

impl Tensor {
    pub fn broadcast_along<A: Into<Axis>>(mut self, axis: A, size: usize) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let dim = axis.to_usize(self.shape.len() + 1);
        self.shape.insert(dim, size);
        self.strides.insert(dim, 0);
        Ok(self)
    }

    pub fn broadcast_like<A: Into<Axis>>(mut self, axis: A, other: &Self) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let tgt_shape = other.shape();
        let dim = axis.to_usize(self.shape.len() + 1);
        self.shape.insert(dim, tgt_shape[dim]);
        self.strides.insert(dim, 0);
        assert_eq!(
            self.shape, tgt_shape,
            "After broadcasting {axis:?}, {:?} does not match {tgt_shape:?}.",
            self.shape
        );
        Ok(self)
    }
}
