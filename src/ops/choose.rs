use crate::tensor::*;

impl Tensor {
    pub fn choose(self, a: Self, b: Self) -> Result<Self, Error> {
        if a.is_same_as(&b) {
            return Ok(a);
        }
        assert_eq!(self.dtype(), Dtype::Boolean);
        assert_eq!(a.dtype(), b.dtype());
        assert_eq!(self.shape(), a.shape());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(self.device(), a.device());
        assert_eq!(a.device(), b.device());
        todo!()
    }
}
