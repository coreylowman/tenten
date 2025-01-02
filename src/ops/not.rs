use std::ops::Not;

use crate::tensor::*;

impl Tensor {
    pub fn not(self) -> Result<Self, Error> {
        assert_eq!(self.dtype(), Dtype::Boolean);
        Ok(self.defer_op("not", |b| b.as_bool().not().into(), "!x"))
    }
}
