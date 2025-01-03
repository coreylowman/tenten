use std::ops::Deref;
use std::{cell::RefCell, rc::Rc};

use cudarc::driver::{DeviceSlice, LaunchAsync};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::tensor::*;
use crate::util::{jit_compile, launch_cfg, CpuIndex};

thread_local! {
    pub(crate) static DEFAULT_DTYPE: RefCell<Dtype> = const {
        RefCell::new(Dtype::Float32)
    }
}

impl Default for Dtype {
    fn default() -> Self {
        DEFAULT_DTYPE.with_borrow(|dtype| *dtype)
    }
}

pub fn set_default_dtype(dtype: Dtype) {
    DEFAULT_DTYPE.with_borrow_mut(|default_dtype| *default_dtype = dtype);
}
pub struct WithDtypeGuard {
    prev: Dtype,
}

pub fn with_dtype(dtype: Dtype) -> WithDtypeGuard {
    WithDtypeGuard {
        prev: DEFAULT_DTYPE.with_borrow_mut(|curr| std::mem::replace(curr, dtype)),
    }
}

impl Drop for WithDtypeGuard {
    fn drop(&mut self) {
        DEFAULT_DTYPE.with_borrow_mut(|x| std::mem::replace(x, self.prev));
    }
}

thread_local! {
    pub(crate) static DEFAULT_DEVICE: RefCell<Device> = const {
        RefCell::new(Device::Cpu)
    }
}

pub fn set_default_device(device: Device) {
    DEFAULT_DEVICE.with_borrow_mut(|default_device| *default_device = device);
}

impl Default for Device {
    fn default() -> Self {
        DEFAULT_DEVICE.with_borrow(|device| *device)
    }
}

pub struct WithDeviceGuard {
    prev: Device,
}

pub fn with_device(device: Device) -> WithDeviceGuard {
    WithDeviceGuard {
        prev: DEFAULT_DEVICE.with_borrow_mut(|curr| std::mem::replace(curr, device)),
    }
}

impl Drop for WithDeviceGuard {
    fn drop(&mut self) {
        DEFAULT_DEVICE.with_borrow_mut(|x| std::mem::replace(x, self.prev));
    }
}

/// ```
/// assert_eq!(tenten::full_strides(&vec![3, 5, 7]), vec![35, 7, 1]);
/// ```
pub fn nd_bytes_strides(shape: &[usize], byte_stride: usize) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    if shape.is_empty() {
        return strides;
    }

    let mut last = byte_stride;
    strides.push(last);

    for n in shape.iter().skip(1).rev() {
        last *= n;
        strides.insert(0, last);
    }

    strides
}

pub fn build_tensor(
    dtype: Dtype,
    shape: Vec<usize>,
    strides: Vec<usize>,
    bytes: BytesPtr,
    requires_grad: bool,
) -> Tensor {
    let lazy = bytes.lazy();
    Tensor {
        id: monotonically_increasing_id(),
        stored_dtype: dtype,
        deferred_dtype: dtype,
        shape: shape.clone(),
        strides: strides.clone(),
        bytes: Rc::new(RefCell::new(bytes)),
        byte_stride: dtype.num_bytes(),
        deferred_ops: Vec::new(),
        gradient: requires_grad.then(|| {
            Box::new(Tensor {
                id: monotonically_increasing_id(),
                stored_dtype: dtype,
                deferred_dtype: dtype,
                shape,
                strides,
                bytes: Rc::new(RefCell::new(lazy)),
                byte_stride: dtype.num_bytes(),
                deferred_ops: Vec::new(),
                gradient: None,
            })
        }),
    }
}

pub fn zeros<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    let numel: usize = shape.iter().product();
    let num_bytes = numel * dtype.num_bytes();
    let bytes = match device {
        Device::Ghost => BytesPtr::Ghost(device, num_bytes),
        Device::Cpu => BytesPtr::Cpu(vec![0; num_bytes]),
        Device::Cuda(dev) => {
            let cuda = crate::util::thread_cuda(dev);
            BytesPtr::Cuda(cuda.alloc_zeros(num_bytes)?)
        }
    };
    Ok(build_tensor(
        dtype,
        shape,
        strides,
        bytes,
        crate::backward::is_recording(),
    ))
}

pub fn zeros_like(x: &Tensor) -> Result<Tensor, Error> {
    let dtype = x.dtype();
    let bytes = match x.bytes.borrow().deref() {
        &BytesPtr::Ghost(Device::Ghost, len) => BytesPtr::Ghost(Device::Ghost, len),
        &BytesPtr::Ghost(Device::Cpu, len) => BytesPtr::Cpu(vec![0; len]),
        &BytesPtr::Ghost(Device::Cuda(ordinal), len) => {
            BytesPtr::Cuda(crate::util::thread_cuda(ordinal).alloc_zeros(len)?)
        }
        BytesPtr::Cpu(src) => BytesPtr::Cpu(vec![0; src.len()]),
        BytesPtr::Cuda(src) => {
            let cuda = src.device();
            BytesPtr::Cuda(cuda.alloc_zeros(src.len())?)
        }
    };
    Ok(build_tensor(
        dtype,
        x.shape.clone(),
        x.strides.clone(),
        bytes,
        x.requires_grad() && crate::backward::is_recording(),
    ))
}

pub unsafe fn empty<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    let numel: usize = shape.iter().product();
    let num_bytes = numel * dtype.num_bytes();
    let bytes = match device {
        Device::Ghost => BytesPtr::Ghost(device, num_bytes),
        Device::Cpu => BytesPtr::Cpu(vec![0; num_bytes]),
        Device::Cuda(dev) => {
            let cuda = crate::util::thread_cuda(dev);
            BytesPtr::Cuda(cuda.alloc(num_bytes)?)
        }
    };
    Ok(build_tensor(
        dtype,
        shape,
        strides,
        bytes,
        crate::backward::is_recording(),
    ))
}

pub unsafe fn empty_like(x: &Tensor) -> Result<Tensor, Error> {
    let dtype = x.dtype();
    let bytes = match x.bytes.borrow().deref() {
        &BytesPtr::Ghost(dev, len) => BytesPtr::Ghost(dev, len),
        BytesPtr::Cpu(src) => BytesPtr::Cpu(vec![0; src.len()]),
        BytesPtr::Cuda(src) => {
            let cuda = src.device();
            BytesPtr::Cuda(cuda.alloc(src.len())?)
        }
    };
    Ok(build_tensor(
        dtype,
        x.shape.clone(),
        x.strides.clone(),
        bytes,
        x.requires_grad() && crate::backward::is_recording(),
    ))
}

pub fn full<Shape, S>(shape: Shape, value: S) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
    S: Into<Scalar>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let value = Into::<Scalar>::into(value);
    let dtype = value.dtype();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    let numel: usize = shape.iter().product();
    let num_bytes = numel * dtype.num_bytes();

    let bytes = match device {
        Device::Ghost => BytesPtr::Ghost(device, num_bytes),
        Device::Cpu => {
            let mut buf = vec![0; num_bytes];
            for i in (0..num_bytes).step_by(dtype.num_bytes()) {
                value.store(&mut buf[i..]);
            }
            BytesPtr::Cpu(buf)
        }
        Device::Cuda(ordinal) => {
            let cuda = crate::util::thread_cuda(ordinal);
            let module_name = std::format!("full{}{}", dtype.short_name(), value.to_string());
            let ty = dtype.cuda_type_name();
            let v = value.to_string();
            if !cuda.has_func(&module_name, "kernel") {
                let kernel_src = std::format!(
                    r#"
extern "C" __global__ void kernel(const size_t *info, uint8_t *buf) {{
    const size_t numel = info[0];
    const size_t byte_stride = info[1];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {{
        *reinterpret_cast<{ty} *>(buf + i * byte_stride) = {v};
    }}
}}
"#
                );
                let ptx = jit_compile(kernel_src)?;
                cuda.load_ptx(ptx, &module_name, &["kernel"])?;
            }
            let mut buf = unsafe { cuda.alloc(num_bytes) }?;
            let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();
            let info = cuda.htod_copy(vec![numel, dtype.num_bytes()])?;
            unsafe { fwd_fn.launch(launch_cfg::<128>(numel as u32), (&info, &mut buf)) }?;

            BytesPtr::Cuda(buf)
        }
    };

    Ok(build_tensor(
        dtype,
        shape,
        strides,
        bytes,
        crate::backward::is_recording(),
    ))
}

pub fn sample_uniform<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    let numel: usize = shape.iter().product();
    let num_bytes = numel * dtype.num_bytes();

    let mut rng = rand::thread_rng();
    let mut init_buf = vec![0; num_bytes];
    for i in (0..num_bytes).step_by(dtype.num_bytes()) {
        let value = match dtype {
            Dtype::Boolean => Scalar::Boolean(rng.gen()),
            Dtype::Float16 => Scalar::Float16(f16::from_f32(rng.gen())),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::from_f32(rng.gen())),
            Dtype::Float32 => Scalar::Float32(rng.gen()),
            Dtype::Float64 => Scalar::Float64(rng.gen()),
            Dtype::Int8 => Scalar::Int8(rng.gen()),
            Dtype::Int16 => Scalar::Int16(rng.gen()),
            Dtype::Int32 => Scalar::Int32(rng.gen()),
            Dtype::Int64 => Scalar::Int64(rng.gen()),
            Dtype::UInt8 => Scalar::UInt8(rng.gen()),
            Dtype::UInt16 => Scalar::UInt16(rng.gen()),
            Dtype::UInt32 => Scalar::UInt32(rng.gen()),
            Dtype::UInt64 => Scalar::UInt64(rng.gen()),
        };
        value.store(&mut init_buf[i..]);
    }

    let bytes = match device {
        Device::Ghost => BytesPtr::Ghost(device, num_bytes),
        Device::Cpu => BytesPtr::Cpu(init_buf),
        Device::Cuda(ordinal) => {
            let cuda = crate::util::thread_cuda(ordinal);
            BytesPtr::Cuda(cuda.htod_copy(init_buf)?)
        }
    };

    Ok(build_tensor(
        dtype,
        shape,
        strides,
        bytes,
        crate::backward::is_recording(),
    ))
}

pub fn sample_uniform_like(x: &Tensor) -> Result<Tensor, Error> {
    let dtype = x.dtype();
    let numel = x.numel();
    let num_bytes = numel * dtype.num_bytes();

    let mut rng = rand::thread_rng();
    let mut init_buf = vec![0; num_bytes];
    for i in (0..num_bytes).step_by(dtype.num_bytes()) {
        let value = match dtype {
            Dtype::Boolean => Scalar::Boolean(rng.gen()),
            Dtype::Float16 => Scalar::Float16(f16::from_f32(rng.gen())),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::from_f32(rng.gen())),
            Dtype::Float32 => Scalar::Float32(rng.gen()),
            Dtype::Float64 => Scalar::Float64(rng.gen()),
            Dtype::Int8 => Scalar::Int8(rng.gen()),
            Dtype::Int16 => Scalar::Int16(rng.gen()),
            Dtype::Int32 => Scalar::Int32(rng.gen()),
            Dtype::Int64 => Scalar::Int64(rng.gen()),
            Dtype::UInt8 => Scalar::UInt8(rng.gen()),
            Dtype::UInt16 => Scalar::UInt16(rng.gen()),
            Dtype::UInt32 => Scalar::UInt32(rng.gen()),
            Dtype::UInt64 => Scalar::UInt64(rng.gen()),
        };
        value.store(&mut init_buf[i..]);
    }

    let bytes = match x.device() {
        Device::Ghost => BytesPtr::Ghost(Device::Ghost, num_bytes),
        Device::Cpu => BytesPtr::Cpu(init_buf),
        Device::Cuda(ordinal) => {
            let cuda = crate::util::thread_cuda(ordinal);
            BytesPtr::Cuda(cuda.htod_copy(init_buf)?)
        }
    };

    Ok(build_tensor(
        dtype,
        x.shape.clone(),
        x.strides.clone(),
        bytes,
        crate::backward::is_recording(),
    ))
}

pub fn sample_normal<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    sample_dist(shape, &StandardNormal)
}

pub fn sample_normal_like(x: &Tensor) -> Result<Tensor, Error> {
    sample_dist_like(x, &StandardNormal)
}

pub fn sample_dist<Shape, D>(shape: Shape, distr: &D) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
    D: Distribution<f32>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    let numel: usize = shape.iter().product();
    let num_bytes = numel * dtype.num_bytes();

    let mut rng = rand::thread_rng();
    let mut init_buf = vec![0; num_bytes];
    for i in (0..num_bytes).step_by(dtype.num_bytes()) {
        let value = match dtype {
            Dtype::Float16 => Scalar::Float16(f16::from_f32(rng.sample(distr))),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::from_f32(rng.sample(distr))),
            Dtype::Float32 => Scalar::Float32(rng.sample(distr)),
            Dtype::Float64 => Scalar::Float64(rng.sample(distr) as f64),
            _ => unimplemented!("Can't sample {dtype:?} values from a f32 distribution"),
        };
        value.store(&mut init_buf[i..]);
    }

    let bytes = match device {
        Device::Ghost => BytesPtr::Ghost(device, num_bytes),
        Device::Cpu => BytesPtr::Cpu(init_buf),
        Device::Cuda(ordinal) => {
            let cuda = crate::util::thread_cuda(ordinal);
            BytesPtr::Cuda(cuda.htod_copy(init_buf)?)
        }
    };

    Ok(build_tensor(
        dtype,
        shape,
        strides,
        bytes,
        crate::backward::is_recording(),
    ))
}

pub fn sample_dist_like<D>(x: &Tensor, distr: &D) -> Result<Tensor, Error>
where
    D: Distribution<f32>,
{
    let dtype = x.dtype();
    let numel = x.numel();
    let num_bytes = numel * dtype.num_bytes();

    let mut rng = rand::thread_rng();
    let mut init_buf = vec![0; num_bytes];
    for i in (0..num_bytes).step_by(dtype.num_bytes()) {
        let value = match dtype {
            Dtype::Float16 => Scalar::Float16(f16::from_f32(rng.sample(distr))),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::from_f32(rng.sample(distr))),
            Dtype::Float32 => Scalar::Float32(rng.sample(distr)),
            Dtype::Float64 => Scalar::Float64(rng.sample(distr) as f64),
            _ => unimplemented!("Can't sample {dtype:?} values from a f32 distribution"),
        };
        value.store(&mut init_buf[i..]);
    }

    let bytes = match x.device() {
        Device::Ghost => BytesPtr::Ghost(Device::Ghost, num_bytes),
        Device::Cpu => BytesPtr::Cpu(init_buf),
        Device::Cuda(ordinal) => {
            let cuda = crate::util::thread_cuda(ordinal);
            BytesPtr::Cuda(cuda.htod_copy(init_buf)?)
        }
    };

    Ok(build_tensor(
        dtype,
        x.shape.clone(),
        x.strides.clone(),
        bytes,
        crate::backward::is_recording(),
    ))
}

pub fn copy_slice<T, Shape>(buf: &[T], shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
    T: Copy + Into<Scalar>,
{
    let dtype = dtype_of::<T>();
    let shape = Into::<Vec<usize>>::into(shape);
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    let numel: usize = shape.iter().product();
    let num_bytes = numel * dtype.num_bytes();

    assert_eq!(
        numel,
        buf.len(),
        "Shape ({shape:?}) has {numel:?} elements, but found {} elements in src slice",
        buf.len()
    );

    let mut init_buf = vec![0; num_bytes];
    for (i, x) in buf.iter().enumerate() {
        let value = Into::<Scalar>::into(*x);
        value.store(&mut init_buf[(i * dtype.num_bytes())..]);
    }

    let bytes = match device {
        Device::Ghost => BytesPtr::Ghost(device, num_bytes),
        Device::Cpu => BytesPtr::Cpu(init_buf),
        Device::Cuda(ordinal) => {
            let cuda = crate::util::thread_cuda(ordinal);
            BytesPtr::Cuda(cuda.htod_copy(init_buf)?)
        }
    };

    Ok(build_tensor(
        dtype,
        shape,
        strides,
        bytes,
        crate::backward::is_recording(),
    ))
}

impl Tensor {
    pub fn into_vec<T>(self) -> Result<Vec<T>, Error>
    where
        T: Copy + From<Scalar>,
    {
        let dtype = self.dtype();
        assert_eq!(dtype, dtype_of::<T>());
        let t = self.undeferred()?.to_device(Device::Cpu)?;
        let mut out = Vec::with_capacity(t.numel());
        match t.bytes.borrow().deref() {
            BytesPtr::Cpu(buf) => {
                let mut idx = CpuIndex::new(&t.shape, &t.strides, t.byte_stride);
                for _ in 0..t.numel() {
                    let i = idx.next().unwrap();
                    let value = dtype.read(&buf[i..]);
                    out.push(value.into())
                }
                assert!(idx.next().is_none());
            }
            _ => unreachable!(),
        };
        Ok(out)
    }
}

impl<T: Copy + Into<Scalar>, const M: usize> From<[T; M]> for Tensor {
    fn from(value: [T; M]) -> Self {
        copy_slice(&value, [M]).unwrap()
    }
}

impl<T: Copy + Into<Scalar>, const M: usize, const N: usize> From<[[T; N]; M]> for Tensor {
    fn from(value: [[T; N]; M]) -> Self {
        copy_slice(
            unsafe { std::slice::from_raw_parts(value.as_ptr() as *const T, M * N) },
            [M, N],
        )
        .unwrap()
    }
}

impl<T: Copy + Into<Scalar>, const M: usize, const N: usize, const O: usize> From<[[[T; O]; N]; M]>
    for Tensor
{
    fn from(value: [[[T; O]; N]; M]) -> Self {
        copy_slice(
            unsafe { std::slice::from_raw_parts(value.as_ptr() as *const T, M * N * O) },
            [M, N, O],
        )
        .unwrap()
    }
}

impl<T: Copy + Into<Scalar>, const M: usize, const N: usize, const O: usize, const P: usize>
    From<[[[[T; P]; O]; N]; M]> for Tensor
{
    fn from(value: [[[[T; P]; O]; N]; M]) -> Self {
        copy_slice(
            unsafe { std::slice::from_raw_parts(value.as_ptr() as *const T, M * N * O * P) },
            [M, N, O, P],
        )
        .unwrap()
    }
}
