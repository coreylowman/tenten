use std::{cell::RefCell, rc::Rc};

use cudarc::driver::DeviceSlice;

use crate::dtype::{Dtype, Scalar};

#[derive(Debug, Clone)]
pub struct Tensor(pub Rc<RefCell<TensorData>>);

#[derive(Debug, Clone)]
pub struct TensorData {
    pub(crate) cur_dtype: Dtype,
    pub(crate) deferred_dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) bytes: BytesPtr, // TODO make this Rc?
    pub(crate) deferred_ops: Vec<DeferredOp>,
    pub(crate) gradient: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct UniqueId(pub(crate) u64);

#[inline(always)]
pub(crate) fn unique_id() -> UniqueId {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    UniqueId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum BytesPtr {
    Phantom,
    Cpu(Vec<u8>),
    Cuda(cudarc::driver::CudaSlice<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    Phantom,
    Cpu,
    Cuda(usize),
}

#[derive(Debug, Clone)]
pub struct DeferredOp {
    pub name: String,
    pub cpu_op: CpuOpPtr,
    pub cuda_op: String,
}

#[derive(Debug, Clone)]
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

impl Tensor {
    pub fn dtype(&self) -> Dtype {
        self.0.borrow().deferred_dtype
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.0.borrow().strides.clone()
    }

    pub fn device(&self) -> Device {
        match &self.0.borrow().bytes {
            BytesPtr::Phantom => Device::Phantom,
            BytesPtr::Cpu(_) => Device::Cpu,
            BytesPtr::Cuda(buf) => Device::Cuda(buf.device().ordinal()),
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.0.borrow().gradient.is_some()
    }

    pub fn numel(&self) -> usize {
        self.0.borrow().shape.iter().product()
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.0.borrow().gradient.clone()
    }

    pub fn alloc(self) -> Result<Self, Error> {
        todo!()
    }

    pub fn get_or_alloc_grad(&self) -> Result<Tensor, Error> {
        let data = self.0.borrow();
        let grad = data
            .gradient
            .clone()
            .expect("Called get_or_alloc_grad on tensor without gradient");
        {
            let mut grad_data = grad.0.borrow_mut();
            let alloc = match &grad_data.bytes {
                BytesPtr::Phantom => true,
                _ => false,
            };
            if alloc {
                grad_data.bytes = match &data.bytes {
                    BytesPtr::Phantom => BytesPtr::Phantom,
                    BytesPtr::Cpu(buf) => BytesPtr::Cpu(vec![0; buf.len()]),
                    BytesPtr::Cuda(buf) => {
                        let cuda = buf.device();
                        BytesPtr::Cuda(cuda.alloc_zeros(buf.len())?)
                    }
                };
            }
        }
        Ok(grad)
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
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
