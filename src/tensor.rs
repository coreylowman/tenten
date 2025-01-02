use std::{cell::RefCell, ops::Deref, rc::Rc};

use cudarc::driver::DeviceSlice;

use crate::dtype::{Dtype, Scalar};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) stored_dtype: Dtype,
    pub(crate) deferred_dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) byte_stride: usize,
    pub(crate) bytes_ptr: Rc<RefCell<BytesPtr>>,
    pub(crate) deferred_ops: Vec<DeferredOp>,
    pub(crate) gradient: Option<Box<Tensor>>,
}

#[derive(Debug, Clone)]
pub struct UniqueId(pub(crate) u64);

#[inline(always)]
pub fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum BytesPtr {
    Phantom,
    Lazy(Device, usize),
    Cpu(Vec<u8>),
    Cuda(cudarc::driver::CudaSlice<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    Phantom,
    Cpu,
    Cuda(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeferredOp {
    pub name: String,
    pub cpu_op: CpuOpPtr,
    pub cuda_op: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CpuOpPtr {
    Simple(fn(&Scalar) -> Scalar),
    WithOptions(fn(&Scalar, &[Scalar]) -> Scalar, Vec<Scalar>),
}

impl From<fn(&Scalar) -> Scalar> for CpuOpPtr {
    fn from(value: fn(&Scalar) -> Scalar) -> Self {
        Self::Simple(value)
    }
}

impl From<(fn(&Scalar, &[Scalar]) -> Scalar, Vec<Scalar>)> for CpuOpPtr {
    fn from((ptr, args): (fn(&Scalar, &[Scalar]) -> Scalar, Vec<Scalar>)) -> Self {
        Self::WithOptions(ptr, args)
    }
}

impl BytesPtr {
    pub fn lazy(&self) -> Self {
        match self {
            BytesPtr::Phantom => BytesPtr::Phantom,
            BytesPtr::Lazy(device, len) => BytesPtr::Lazy(*device, *len),
            BytesPtr::Cpu(vec) => BytesPtr::Lazy(Device::Cpu, vec.len()),
            BytesPtr::Cuda(cuda_slice) => BytesPtr::Lazy(
                Device::Cuda(cuda_slice.device().ordinal()),
                cuda_slice.len(),
            ),
        }
    }
}

impl Tensor {
    pub fn dtype(&self) -> Dtype {
        self.deferred_dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn device(&self) -> Device {
        match self.bytes_ptr.borrow().deref() {
            BytesPtr::Phantom => Device::Phantom,
            BytesPtr::Lazy(device, _) => *device,
            BytesPtr::Cpu(_) => Device::Cpu,
            BytesPtr::Cuda(buf) => Device::Cuda(buf.device().ordinal()),
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.gradient.is_some()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn num_dims(&self) -> usize {
        self.shape.len()
    }

    pub fn grad(&self) -> Option<Tensor> {
        // here we are cloning deferred ops (which will be empty), shape/strides, dtype. the bytes_ptr is behid a rc so its cheap
        self.gradient.clone().map(|t| *t)
    }

    pub fn alloc(self) -> Result<Self, Error> {
        let maybe_alloc = match self.bytes_ptr.borrow().deref() {
            &BytesPtr::Lazy(device, len) => Some((device, len)),
            _ => None,
        };
        let (device, len) = match maybe_alloc {
            Some((d, l)) => (d, l),
            None => return Ok(self),
        };
        *self.bytes_ptr.borrow_mut() = match device {
            Device::Phantom => BytesPtr::Phantom,
            Device::Cpu => BytesPtr::Cpu(vec![0u8; len]),
            Device::Cuda(ordinal) => {
                let cuda = crate::cuda::thread_cuda(ordinal);
                BytesPtr::Cuda(cuda.alloc_zeros(len)?)
            }
        };
        Ok(self)
    }

    pub fn is_same_as(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.bytes_ptr, &other.bytes_ptr) && self.deferred_ops == other.deferred_ops
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    CudaDriverError(cudarc::driver::DriverError),
    CublasError(cudarc::cublas::result::CublasError),
    #[cfg(feature = "cudnn")]
    CudnnError(cudarc::cudnn::CudnnError),
}
