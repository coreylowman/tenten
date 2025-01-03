use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::{
    cublas::result::CublasError,
    driver::{CudaDevice, DriverError},
    nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx},
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

pub fn launch_cfg<const NUM_THREADS: u32>(n: u32) -> cudarc::driver::LaunchConfig {
    let num_blocks = n.div_ceil(NUM_THREADS);
    cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct CpuIndex<'a> {
    pub(crate) indices: Vec<usize>,
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) next: Option<usize>,
    pub(crate) contiguous: Option<usize>,
    pub(crate) byte_stride: usize,
}

impl<'a> CpuIndex<'a> {
    #[inline]
    pub fn new(shape: &'a [usize], strides: &'a [usize], byte_stride: usize) -> Self {
        Self {
            indices: vec![0; shape.len()],
            shape,
            strides,
            next: if !shape.is_empty() { Some(0) } else { None },
            contiguous: (strides == crate::init::nd_bytes_strides(shape, byte_stride))
                .then(|| shape.iter().product::<usize>()),
            byte_stride,
        }
    }
}

impl CpuIndex<'_> {
    #[inline(always)]
    pub fn next(&mut self) -> Option<usize> {
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
                    Some(idx * self.byte_stride)
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
pub fn thread_cuda(ordinal: usize) -> Arc<CudaDevice> {
    CUDA_INSTANCES.with(|t| {
        let mut instances = t.borrow_mut();
        instances.get(&ordinal).cloned().unwrap_or_else(|| {
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

pub fn find_cuda_arch() -> &'static String {
    static CUDA_ARCH: OnceLock<String> = OnceLock::new();
    CUDA_ARCH.get_or_init(|| {
        if let Ok(arch) = std::env::var("CUDA_ARCH") {
            arch
        } else {
            let smi_compute_cap = {
                let out = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=compute_cap")
                    .arg("--format=csv,noheader")
                    .output()
                    .expect("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.");
                let out = std::str::from_utf8(&out.stdout).unwrap();
                out.lines().next().unwrap().replace('.', "").parse::<usize>().unwrap()
            };

            let max_nvcc_code = {
                let out = std::process::Command::new("nvcc")
                    .arg("--list-gpu-code")
                    .output()
                    .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
                let out = std::str::from_utf8(&out.stdout).unwrap();

                let out = out.lines().collect::<Vec<&str>>();
                let mut codes = Vec::with_capacity(out.len());
                for code in out {
                    let code = code.replace('a', "");
                    codes.push(code.split("_").last().unwrap().parse::<usize>().unwrap());
                }
                codes.sort();
                if !codes.contains(&smi_compute_cap) {
                    panic!("nvcc cannot target gpu arch {smi_compute_cap}. Available nvcc targets are {codes:?}.");
                }
                *codes.last().unwrap()
            };

            let arch = smi_compute_cap.min(max_nvcc_code);

            std::format!("sm_{arch}")
        }
    })
}

pub fn find_cuda_include_dir() -> &'static String {
    static CUDA_INCLUDE_DIR: OnceLock<String> = OnceLock::new();
    CUDA_INCLUDE_DIR.get_or_init(|| {
        use std::path::PathBuf;

        let env_vars = [
            "CUDA_PATH",
            "CUDA_ROOT",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CUDNN_LIB",
        ];
        let env_vars = env_vars
            .into_iter()
            .map(std::env::var)
            .filter_map(Result::ok)
            .map(Into::<PathBuf>::into);

        let roots = [
            "/usr",
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
            "C:/Program Files/NVIDIA GPU Computing Toolkit",
            "C:/CUDA",
        ];
        let roots = roots.into_iter().map(Into::<PathBuf>::into);

        let root = env_vars
            .chain(roots)
            .find(|path| path.join("include").join("cuda.h").is_file())
            .unwrap();

        std::format!("{}", root.join("include").display())
    })
}

pub fn jit_compile(mut src: String) -> Result<Ptx, Error> {
    let arch = find_cuda_arch();
    let cuda_include_dir = find_cuda_include_dir();
    src.insert_str(
        0,
        "
#include \"cuda_fp16.h\"
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long int64_t;
",
    );
    let ptx = compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some(arch),
            include_paths: vec![cuda_include_dir.clone()],
            ..Default::default()
        },
    )?;
    Ok(ptx)
}

impl From<cudarc::nvrtc::CompileError> for Error {
    fn from(value: cudarc::nvrtc::CompileError) -> Self {
        Self::CompilerError(value)
    }
}
