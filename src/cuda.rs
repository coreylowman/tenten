use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::{
    cublas::{result::CublasError, CudaBlas},
    driver::{CudaDevice, CudaSlice, CudaStream, DriverError},
};

#[cfg(feature = "cudnn")]
use cudarc::cudnn::Cudnn;

use crate::tensor::Error;

pub(crate) fn launch_cfg<const NUM_THREADS: u32>(n: u32) -> cudarc::driver::LaunchConfig {
    let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

thread_local!(pub static CUDA_INSTANCES: RefCell<HashMap<usize, Arc<CudaDevice>>> = RefCell::new(HashMap::new()));

#[inline(always)]
pub(crate) fn thread_cuda(ordinal: usize) -> Arc<CudaDevice> {
    CUDA_INSTANCES.with(|t| {
        let mut instances = t.borrow_mut();
        instances
            .get(&ordinal)
            .map(|v| v.clone())
            .unwrap_or_else(|| {
                let new_instance = CudaDevice::new(ordinal).unwrap();
                instances.insert(ordinal, new_instance.clone());
                new_instance
            })
    })
}

impl From<CublasError> for Error {
    fn from(value: CublasError) -> Self {
        Self::CublasError(value)
    }
}

impl From<DriverError> for Error {
    fn from(value: DriverError) -> Self {
        Self::CudaDriverError(value)
    }
}

#[cfg(feature = "cudnn")]
impl From<cudarc::cudnn::CudnnError> for Error {
    fn from(value: cudarc::cudnn::CudnnError) -> Self {
        Self::CudnnError(value)
    }
}
