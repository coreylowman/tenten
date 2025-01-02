use crate::tensor::{Error, Tensor};

impl Tensor {
    pub fn sub_assign(&mut self, other: Self) -> Result<(), Error> {
        let _no_grad = crate::backward::no_grad();
        if self.is_same_as(&other) {
            self.fill_with_zeros()
        } else {
            self.add_assign(other.negate()?)
        }
    }
}
