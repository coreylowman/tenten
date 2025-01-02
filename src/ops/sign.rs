use crate::tensor::{Error, Tensor};

impl Tensor {
    pub fn sign(self) -> Result<Self, Error> {
        // TODO can optimize this using replace code directly so we don't have to do 2x replace
        let dtype = self.dtype();
        let pos_mask = self.clone().gt_scalar(dtype.zero())?;
        let neg_mask = self.clone().lt_scalar(dtype.zero())?;
        self.replace(pos_mask, dtype.one())?
            .replace(neg_mask, dtype.one().negate())
    }
}
