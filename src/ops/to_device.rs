use crate::tensor::*;

impl Tensor {
    pub fn to_device(mut self, device: Device) -> Result<Self, Error> {
        // TODO can keep deferred ops and just send existing data to the device
        // TODO I think we need to do a Rc::make_mut
        todo!()
    }
}
