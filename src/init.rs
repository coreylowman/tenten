use std::ops::Deref;
use std::{cell::RefCell, rc::Rc};

use cudarc::driver::DeviceSlice;

use crate::dtype::*;
use crate::tensor::*;

thread_local! {
    pub(crate) static DEFAULT_DTYPE: RefCell<Dtype> = const {
        RefCell::new(Dtype::Float32)
    }
}

pub fn set_default_dtype(dtype: Dtype) {
    DEFAULT_DTYPE.with_borrow_mut(|default_dtype| *default_dtype = dtype);
}

impl Default for Dtype {
    fn default() -> Self {
        DEFAULT_DTYPE.with_borrow(|dtype| *dtype)
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

/// ```
/// assert_eq!(tenten::full_strides(&vec![3, 5, 7]), vec![35, 7, 1]);
/// ```
pub fn nd_bytes_strides(shape: &[usize], byte_stride: usize) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    if shape.len() == 0 {
        return strides;
    }

    let mut last = byte_stride;
    strides.push(1);

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
        stored_dtype: dtype,
        deferred_dtype: dtype,
        shape: shape.clone(),
        strides: strides.clone(),
        bytes_ptr: Rc::new(RefCell::new(bytes)),
        byte_stride: dtype.num_bytes(),
        deferred_ops: Vec::new(),
        gradient: requires_grad.then(|| {
            Box::new(Tensor {
                stored_dtype: dtype,
                deferred_dtype: dtype,
                shape,
                strides,
                bytes_ptr: Rc::new(RefCell::new(lazy)),
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
    let bytes = match device {
        Device::Phantom => BytesPtr::Phantom,
        Device::Cpu => BytesPtr::Cpu(vec![0; numel]),
        Device::Cuda(dev) => {
            let cuda = crate::cuda::thread_cuda(dev);
            BytesPtr::Cuda(cuda.alloc_zeros(numel)?)
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
    pub fn zeros_like(&self) -> Result<Self, Error> {
        let dtype = self.deferred_dtype;
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let bytes = match self.bytes_ptr.borrow().deref() {
            BytesPtr::Phantom => BytesPtr::Phantom,
            &BytesPtr::Lazy(Device::Phantom, _) => BytesPtr::Phantom,
            &BytesPtr::Lazy(Device::Cpu, len) => BytesPtr::Cpu(vec![0; len]),
            &BytesPtr::Lazy(Device::Cuda(ordinal), len) => {
                BytesPtr::Cuda(crate::cuda::thread_cuda(ordinal).alloc_zeros(len)?)
            }
            BytesPtr::Cpu(src) => BytesPtr::Cpu(vec![0; src.len()]),
            BytesPtr::Cuda(src) => {
                let cuda = src.device();
                BytesPtr::Cuda(cuda.alloc_zeros(src.len())?)
            }
        };
        Ok(build_tensor(
            dtype,
            shape,
            strides,
            bytes,
            self.requires_grad() && crate::backward::is_recording(),
        ))
    }
}

pub fn ones<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    todo!()
}

impl Tensor {
    pub fn ones_like(&self) -> Result<Self, Error> {
        todo!()
    }
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
    let bytes = match device {
        Device::Phantom => BytesPtr::Phantom,
        Device::Cpu => BytesPtr::Cpu(vec![0; numel]),
        Device::Cuda(dev) => {
            let cuda = crate::cuda::thread_cuda(dev);
            BytesPtr::Cuda(cuda.alloc(numel)?)
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
    pub unsafe fn empty_like(&self) -> Result<Self, Error> {
        let dtype = self.deferred_dtype;
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let bytes = match self.bytes_ptr.borrow().deref() {
            BytesPtr::Phantom => BytesPtr::Phantom,
            BytesPtr::Lazy(dev, len) => todo!(),
            BytesPtr::Cpu(src) => BytesPtr::Cpu(vec![0; src.len()]),
            BytesPtr::Cuda(src) => {
                let cuda = src.device();
                BytesPtr::Cuda(cuda.alloc(src.len())?)
            }
        };
        Ok(build_tensor(
            dtype,
            shape,
            strides,
            bytes,
            self.requires_grad() && crate::backward::is_recording(),
        ))
    }
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
    todo!()
}

pub fn sample_uniform<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    todo!()
}

impl Tensor {
    pub fn sample_uniform_like(&self) -> Self {
        todo!()
    }
}

pub fn sample_normal<Shape>(shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let shape = Into::<Vec<usize>>::into(shape);
    let dtype: Dtype = Default::default();
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    todo!()
}

impl Tensor {
    pub fn sample_normal_like(&self) -> Result<Self, Error> {
        todo!()
    }
}

pub fn dtype_of<T>() -> Dtype {
    match std::any::type_name::<T>() {
        "half::f16" => Dtype::Float16,
        "half::bf16" => Dtype::BFloat16,
        "f32" => Dtype::Float32,
        "f64" => Dtype::Float64,
        "i8" => Dtype::Int8,
        "i16" => Dtype::Int16,
        "i32" => Dtype::Int32,
        "i64" => Dtype::Int64,
        "u8" => Dtype::UInt8,
        "u16" => Dtype::UInt16,
        "u32" => Dtype::UInt32,
        "u64" => Dtype::UInt64,
        not_supported => unimplemented!("unable to handle type {not_supported}"),
    }
}

pub fn copy_slice<T, Shape>(buf: &[T], shape: Shape) -> Result<Tensor, Error>
where
    Shape: Into<Vec<usize>>,
{
    let dtype = dtype_of::<T>();
    let shape = Into::<Vec<usize>>::into(shape);
    let strides = nd_bytes_strides(&shape, dtype.num_bytes());
    let device: Device = Default::default();
    todo!()
}

impl<T: Into<Scalar>, const M: usize> From<[T; M]> for Tensor {
    fn from(value: [T; M]) -> Self {
        let dtype = dtype_of::<T>();
        let shape = vec![M];
        let strides = nd_bytes_strides(&shape, dtype.num_bytes());
        let device: Device = Default::default();
        todo!()
    }
}

impl<T: Into<Scalar>, const M: usize, const N: usize> From<[[T; N]; M]> for Tensor {
    fn from(value: [[T; N]; M]) -> Self {
        let dtype = dtype_of::<T>();
        let shape = vec![M, N];
        let strides = nd_bytes_strides(&shape, dtype.num_bytes());
        let device: Device = Default::default();
        todo!()
    }
}

impl<T: Into<Scalar>, const M: usize, const N: usize, const O: usize> From<[[[T; O]; N]; M]>
    for Tensor
{
    fn from(value: [[[T; O]; N]; M]) -> Self {
        let dtype = dtype_of::<T>();
        let shape = vec![M, N, O];
        let strides = nd_bytes_strides(&shape, dtype.num_bytes());
        let device: Device = Default::default();
        todo!()
    }
}

impl<T: Into<Scalar>, const M: usize, const N: usize, const O: usize, const P: usize>
    From<[[[[T; P]; O]; N]; M]> for Tensor
{
    fn from(value: [[[[T; P]; O]; N]; M]) -> Self {
        let dtype = dtype_of::<T>();
        let shape = vec![M, N, O, P];
        let strides = nd_bytes_strides(&shape, dtype.num_bytes());
        let device: Device = Default::default();
        todo!()
    }
}
