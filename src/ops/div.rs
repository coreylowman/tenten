use crate::tensor::{Dtype, Error, Tensor};

impl Tensor {
    pub fn div(mut self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            self.fill_with(self.dtype().one())?;
            Ok(self)
        } else {
            match self.dtype() {
                Dtype::Float16 | Dtype::BFloat16 | Dtype::Float32 | Dtype::Float64 => {
                    self.mul(other.recip()?)
                }
                _ => todo!(),
            }
        }
    }
}
