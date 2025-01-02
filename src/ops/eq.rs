use crate::tensor::*;

impl Tensor {
    pub fn eq(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            return crate::init::full(self.shape(), true)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("mutate dtype")
    }
}
