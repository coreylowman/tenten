use crate::tensor::*;

impl Tensor {
    pub fn min(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            Ok(self)
        } else {
            // TODO this results in 2 kernels being launched, optimize with a special kernel
            self.clone().le(other.clone())?.choose(self, other)
        }
    }
}
