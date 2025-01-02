use crate::tensor::*;

pub trait ConcatAlong {
    fn concat_along<A: Into<Axis>>(self, axis: A) -> Result<Tensor, Error>;
}

impl<Tensors: Into<Vec<Tensor>>> ConcatAlong for Tensors {
    fn concat_along<A: Into<Axis>>(self, axis: A) -> Result<Tensor, Error> {
        let tensors = Into::<Vec<Tensor>>::into(self);
        todo!()
    }
}
