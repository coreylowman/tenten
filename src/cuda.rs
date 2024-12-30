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

#[derive(Debug, Clone)]
pub(crate) struct ThreadLocalCuda {
    pub(crate) dev: Arc<CudaDevice>,
    pub(crate) blas: Arc<CudaBlas>,
    #[cfg(feature = "cudnn")]
    #[allow(unused)]
    pub(crate) cudnn: Arc<Cudnn>,
    /// A second stream for kernels to optionally execute on.
    pub(crate) par_stream: Arc<CudaStream>,
    pub(crate) workspace: Arc<Mutex<CudaSlice<u8>>>,
}

impl ThreadLocalCuda {
    fn try_new(ordinal: usize) -> Result<Self, Error> {
        let dev = CudaDevice::new(ordinal)?;
        let blas = Arc::new(CudaBlas::new(dev.clone())?);
        #[cfg(feature = "cudnn")]
        let cudnn = Cudnn::new(dev.clone())?;
        let par_stream = Arc::new(dev.fork_default_stream()?);
        let workspace = Arc::new(Mutex::new(dev.alloc_zeros::<u8>(1)?));
        Ok(Self {
            dev,
            blas,
            #[cfg(feature = "cudnn")]
            cudnn,
            par_stream,
            workspace,
        })
    }
}

thread_local!(pub static CUDA_INSTANCES: RefCell<HashMap<usize, ThreadLocalCuda>> = RefCell::new(HashMap::new()));

#[inline(always)]
pub(crate) fn thread_cuda(ordinal: usize) -> ThreadLocalCuda {
    CUDA_INSTANCES.with(|t| {
        let mut instances = t.borrow_mut();
        instances
            .get(&ordinal)
            .map(|v| v.clone())
            .unwrap_or_else(|| {
                let new_instance = ThreadLocalCuda::try_new(ordinal).unwrap();
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
