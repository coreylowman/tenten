use crate::tensor::*;

pub trait Stack {
    fn stack(self) -> Result<Tensor, Error>;
}

impl<Tensors: Into<Vec<Tensor>>> Stack for Tensors {
    fn stack(self) -> Result<Tensor, Error> {
        let tensors = Into::<Vec<Tensor>>::into(self);
        todo!()
    }
}
