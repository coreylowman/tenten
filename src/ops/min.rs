use crate::tensor::*;

impl Tensor {
    pub fn min(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            Ok(self)
        } else {
            // TODO we can optimize this with a special kernel
            self.clone().le(other.clone())?.choose(self, other)
        }
    }
}
