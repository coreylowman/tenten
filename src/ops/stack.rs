use crate::tensor::*;

pub fn stack<T>(tensors: T) -> Result<Tensor, Error>
where
    T: Into<Vec<Tensor>>,
{
    todo!()
}
