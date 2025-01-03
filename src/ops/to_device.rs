use std::{ops::Deref, rc::Rc};

use crate::{tensor::*, util::thread_cuda};

impl Tensor {
    pub fn to_device(mut self, device: Device) -> Result<Self, Error> {
        if self.device() == device {
            Ok(self)
        } else {
            let bytes = Rc::make_mut(&mut self.bytes);

            *bytes.borrow_mut() = match bytes.borrow().deref() {
                BytesPtr::Phantom => todo!(),
                &BytesPtr::Lazy(_, len) => BytesPtr::Lazy(device, len),
                BytesPtr::Cpu(buf) => match device {
                    Device::Phantom => BytesPtr::Phantom,
                    Device::Cuda(ordinal) => {
                        let cuda = thread_cuda(ordinal);
                        BytesPtr::Cuda(cuda.htod_sync_copy(buf)?)
                    }
                    _ => unreachable!(),
                },
                BytesPtr::Cuda(buf) => match device {
                    Device::Phantom => BytesPtr::Phantom,
                    Device::Cpu => BytesPtr::Cpu(buf.device().dtoh_sync_copy(buf)?),
                    Device::Cuda(_) => {
                        todo!()
                    }
                },
            };

            Ok(self)
        }
    }
}
