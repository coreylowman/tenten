use crate::tensor::*;

impl Tensor {
    pub fn fill_with_ones(&self) -> Result<(), Error> {
        // TODO clear deferred ops
        // TODO add grad backward which sets grad to zero
        todo!()
    }
}
