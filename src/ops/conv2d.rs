use crate::tensor::*;

impl Tensor {
    pub fn conv2d(
        self,
        weight: Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Self, Error> {
        assert_eq!(self.dtype(), weight.dtype());
        todo!()
    }
}
