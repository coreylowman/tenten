use crate::tensor::{Error, Tensor};

impl Tensor {
    /// TODO feature gate this
    pub fn gelu_true(self) -> Result<Self, Error> {
        todo!("defer_op")
    }
}
