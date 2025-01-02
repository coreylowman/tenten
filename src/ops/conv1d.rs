use crate::tensor::*;

impl Tensor {
    pub fn conv1d(
        self,
        weight: Tensor,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self, Error> {
        assert_eq!(self.dtype(), weight.dtype());
        todo!()
    }
}
