use crate::tensor::*;

impl Tensor {
    pub fn sign(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        self.clone()
            .gt_scalar(dtype.zero())?
            .sub(self.lt_scalar(dtype.zero())?)
    }
}
