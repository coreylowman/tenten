use crate::tensor::*;

impl Tensor {
    pub fn sub(mut self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            self.fill_with_zeros()?;
            Ok(self)
        } else {
            self.add(other.negate()?)
        }
    }
}
