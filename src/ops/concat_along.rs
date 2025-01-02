use crate::tensor::*;

pub fn concat_along<T, A>(tensors: T, axis: A) -> Result<Tensor, Error>
where
    T: Into<Vec<Tensor>>,
    A: Into<Axis>,
{
    todo!()
}
