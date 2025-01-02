use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::{
    cublas::result::CublasError,
    driver::{CudaDevice, DriverError},
};

#[cfg(feature = "cudnn")]
use cudarc::cudnn::Cudnn;

use crate::tensor::Error;

pub fn all_some<T, const N: usize>(arr: [Option<T>; N]) -> Option<[T; N]> {
    if arr.iter().all(Option::is_some) {
        Some(arr.map(Option::unwrap))
    } else {
        None
    }
}

pub(crate) fn launch_cfg<const NUM_THREADS: u32>(n: u32) -> cudarc::driver::LaunchConfig {
    let num_blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CpuIndex<'a> {
    pub(crate) indices: Vec<usize>,
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) next: Option<usize>,
    pub(crate) contiguous: Option<usize>,
}

impl<'a> CpuIndex<'a> {
    #[inline]
    pub(crate) fn new(shape: &'a [usize], strides: &'a [usize], byte_stride: usize) -> Self {
        Self {
            indices: Default::default(),
            shape,
            strides,
            next: if shape.len() > 0 { Some(0) } else { None },
            contiguous: (strides == &crate::init::nd_bytes_strides(shape, byte_stride))
                .then(|| shape.iter().product::<usize>()),
        }
    }
}

impl<'a> CpuIndex<'a> {
    pub(crate) fn get_strided_index(&self, mut idx: usize) -> usize {
        let mut out = 0;

        for (dim, stride) in self.shape.iter().zip(self.strides.iter()).rev() {
            out += (idx % dim) * stride;
            idx /= dim;
        }

        out
    }

    #[inline(always)]
    pub(crate) fn next(&mut self) -> Option<usize> {
        match self.contiguous {
            Some(numel) => match self.next.as_mut() {
                Some(i) => {
                    let idx = *i;
                    let next = idx + 1;
                    if next >= numel {
                        self.next = None;
                    } else {
                        *i = next;
                    }
                    Some(idx)
                }
                None => None,
            },
            None => self.next_with_idx().map(|(i, _)| i),
        }
    }

    #[inline(always)]
    pub(crate) fn next_with_idx(&mut self) -> Option<(usize, Vec<usize>)> {
        match (self.shape.len(), self.next.as_mut()) {
            (_, None) => None,
            (0, Some(i)) => {
                let idx = (*i, self.indices.clone());
                self.next = None;
                Some(idx)
            }
            (_, Some(i)) => {
                let idx = (*i, self.indices.clone());
                let mut dim = self.shape.len() - 1;
                loop {
                    self.indices[dim] += 1;
                    *i += self.strides[dim];

                    if self.indices[dim] < self.shape[dim] {
                        break;
                    }

                    *i -= self.shape[dim] * self.strides[dim];
                    self.indices[dim] = 0;

                    if dim == 0 {
                        self.next = None;
                        break;
                    }

                    dim -= 1;
                }
                Some(idx)
            }
        }
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
