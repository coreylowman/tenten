use crate::tensor::{Error, Tensor};

impl Tensor {
    pub fn sub_assign(&mut self, other: Self) -> Result<(), Error> {
        if self.is_same_as(&other) {
            self.fill_with_zeros()
        } else {
            self.add_assign(other.negate()?)
        }
    }
}
