use crate::tensor::*;

impl Tensor {
    pub fn max(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            Ok(self)
        } else {
            // TODO this results in multiple kernels being executed. optimize with a special kernel
            self.clone().ge(other.clone())?.choose(self, other)
        }
    }
}
