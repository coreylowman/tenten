use crate::tensor::{Error, Tensor};

impl Tensor {
    pub fn div(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            self.fill_with_ones()?;
            Ok(self)
        } else {
            // TODO handle integer values
            self.mul(other.recip()?)
        }
    }
}
