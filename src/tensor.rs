pub use half::{bf16, f16};

use std::{cell::RefCell, ops::Deref, rc::Rc};

use cudarc::driver::DeviceSlice;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: MonotonicallyIncreasingId,
    pub stored_dtype: Dtype,
    pub deferred_dtype: Dtype,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub byte_stride: usize,
    pub bytes: Rc<RefCell<BytesPtr>>,
    pub deferred_ops: Vec<DeferredOp>,
    pub gradient: Option<Box<Tensor>>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicallyIncreasingId(pub(crate) u64);

#[inline(always)]
pub fn monotonically_increasing_id() -> MonotonicallyIncreasingId {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    MonotonicallyIncreasingId(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum BytesPtr {
    Ghost(Device, usize),
    Cpu(Vec<u8>),
    Cuda(cudarc::driver::CudaSlice<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    Ghost,
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
            BytesPtr::Ghost(device, len) => BytesPtr::Ghost(*device, *len),
            BytesPtr::Cpu(vec) => BytesPtr::Ghost(Device::Cpu, vec.len()),
            BytesPtr::Cuda(cuda_slice) => BytesPtr::Ghost(
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
        match self.bytes.borrow().deref() {
            BytesPtr::Ghost(device, _) => *device,
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

    pub fn set_new_grad(&mut self) -> Option<Tensor> {
        if let Some(grad) = self.gradient.as_mut() {
            let bytes_ptr = self.bytes.borrow().lazy();
            *grad = Box::new(Tensor {
                id: monotonically_increasing_id(),
                stored_dtype: self.deferred_dtype,
                deferred_dtype: self.deferred_dtype,
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                byte_stride: self.byte_stride,
                bytes: Rc::new(RefCell::new(bytes_ptr)),
                deferred_ops: Vec::new(),
                gradient: None,
            })
        }
        self.grad()
    }

    pub fn alloc(self) -> Result<Self, Error> {
        let maybe_alloc = match self.bytes.borrow().deref() {
            &BytesPtr::Ghost(device, len) => Some((device, len)),
            _ => None,
        };
        let (device, len) = match maybe_alloc {
            Some((d, l)) => (d, l),
            None => return Ok(self),
        };
        *self.bytes.borrow_mut() = match device {
            Device::Ghost => BytesPtr::Ghost(device, len),
            Device::Cpu => BytesPtr::Cpu(vec![0u8; len]),
            Device::Cuda(ordinal) => {
                let cuda = crate::util::thread_cuda(ordinal);
                BytesPtr::Cuda(cuda.alloc_zeros(len)?)
            }
        };
        Ok(self)
    }

    pub fn is_same_as(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.bytes, &other.bytes) && self.deferred_ops == other.deferred_ops
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

#[non_exhaustive]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Boolean,
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl Dtype {
    #[inline(always)]
    pub fn short_name(&self) -> &str {
        match self {
            Dtype::Boolean => "bool",
            Dtype::Float16 => "f16",
            Dtype::BFloat16 => "bf16",
            Dtype::Float32 => "f32",
            Dtype::Float64 => "f64",
            Dtype::Int8 => "i8",
            Dtype::Int16 => "i16",
            Dtype::Int32 => "i32",
            Dtype::Int64 => "i64",
            Dtype::UInt8 => "u8",
            Dtype::UInt16 => "u16",
            Dtype::UInt32 => "u32",
            Dtype::UInt64 => "u64",
        }
    }

    #[inline(always)]
    pub fn cuda_type_name(&self) -> &str {
        match self {
            Dtype::Boolean => "bool",
            Dtype::Float16 => "__half",
            Dtype::BFloat16 => "__nv_bfloat16",
            Dtype::Float32 => "float",
            Dtype::Float64 => "double",
            Dtype::Int8 => "int8_t",
            Dtype::Int16 => "int16_t",
            Dtype::Int32 => "int32_t",
            Dtype::Int64 => "int64_t",
            Dtype::UInt8 => "uint8_t",
            Dtype::UInt16 => "uint16_t",
            Dtype::UInt32 => "uint32_t",
            Dtype::UInt64 => "uint64_t",
        }
    }

    #[inline(always)]
    pub fn num_bytes(&self) -> usize {
        match self {
            Dtype::Boolean => 1,
            Dtype::Float16 => 2,
            Dtype::BFloat16 => 2,
            Dtype::Float32 => 4,
            Dtype::Float64 => 8,
            Dtype::Int8 => 1,
            Dtype::Int16 => 2,
            Dtype::Int32 => 4,
            Dtype::Int64 => 8,
            Dtype::UInt8 => 1,
            Dtype::UInt16 => 2,
            Dtype::UInt32 => 4,
            Dtype::UInt64 => 8,
        }
    }

    #[inline(always)]
    pub fn read(&self, buf: &[u8]) -> Scalar {
        match self {
            Dtype::Boolean => Scalar::Boolean(buf[0] == 1),
            Dtype::Float16 => Scalar::Float16(f16::from_le_bytes([buf[0], buf[1]])),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::from_le_bytes([buf[0], buf[1]])),
            Dtype::Float32 => Scalar::Float32(f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
            Dtype::Float64 => Scalar::Float64(f64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ])),
            Dtype::Int8 => Scalar::Int8(i8::from_le_bytes([buf[0]])),
            Dtype::Int16 => Scalar::Int16(i16::from_le_bytes([buf[0], buf[1]])),
            Dtype::Int32 => Scalar::Int32(i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
            Dtype::Int64 => Scalar::Int64(i64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ])),
            Dtype::UInt8 => Scalar::UInt8(buf[0]),
            Dtype::UInt16 => Scalar::UInt16(u16::from_le_bytes([buf[0], buf[1]])),
            Dtype::UInt32 => Scalar::UInt32(u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])),
            Dtype::UInt64 => Scalar::UInt64(u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ])),
        }
    }

    #[inline(always)]
    pub fn zero(&self) -> Scalar {
        match self {
            Dtype::Boolean => Scalar::Boolean(false),
            Dtype::Float16 => Scalar::Float16(f16::ZERO),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::ZERO),
            Dtype::Float32 => Scalar::Float32(0.0),
            Dtype::Float64 => Scalar::Float64(0.0),
            Dtype::Int8 => Scalar::Int8(0),
            Dtype::Int16 => Scalar::Int16(0),
            Dtype::Int32 => Scalar::Int32(0),
            Dtype::Int64 => Scalar::Int64(0),
            Dtype::UInt8 => Scalar::UInt8(0),
            Dtype::UInt16 => Scalar::UInt16(0),
            Dtype::UInt32 => Scalar::UInt32(0),
            Dtype::UInt64 => Scalar::UInt64(0),
        }
    }

    #[inline(always)]
    pub fn one(&self) -> Scalar {
        match self {
            Dtype::Boolean => Scalar::Boolean(true),
            Dtype::Float16 => Scalar::Float16(f16::ONE),
            Dtype::BFloat16 => Scalar::BFloat16(bf16::ONE),
            Dtype::Float32 => Scalar::Float32(1.0),
            Dtype::Float64 => Scalar::Float64(1.0),
            Dtype::Int8 => Scalar::Int8(1),
            Dtype::Int16 => Scalar::Int16(1),
            Dtype::Int32 => Scalar::Int32(1),
            Dtype::Int64 => Scalar::Int64(1),
            Dtype::UInt8 => Scalar::UInt8(1),
            Dtype::UInt16 => Scalar::UInt16(1),
            Dtype::UInt32 => Scalar::UInt32(1),
            Dtype::UInt64 => Scalar::UInt64(1),
        }
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum Scalar {
    Boolean(bool),
    Float16(half::f16),
    BFloat16(half::bf16),
    Float32(f32),
    Float64(f64),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
}

impl Scalar {
    #[inline(always)]
    pub fn to_string(&self) -> String {
        match self {
            Scalar::Boolean(x) => x.to_string(),
            Scalar::Float16(x) => x.to_string(),
            Scalar::BFloat16(x) => x.to_string(),
            Scalar::Float32(x) => x.to_string(),
            Scalar::Float64(x) => x.to_string(),
            Scalar::Int8(x) => x.to_string(),
            Scalar::Int16(x) => x.to_string(),
            Scalar::Int32(x) => x.to_string(),
            Scalar::Int64(x) => x.to_string(),
            Scalar::UInt8(x) => x.to_string(),
            Scalar::UInt16(x) => x.to_string(),
            Scalar::UInt32(x) => x.to_string(),
            Scalar::UInt64(x) => x.to_string(),
        }
    }
}

impl Scalar {
    #[inline(always)]
    pub fn dtype(&self) -> Dtype {
        match self {
            Scalar::Boolean(_) => Dtype::Boolean,
            Scalar::Float16(_) => Dtype::Float16,
            Scalar::BFloat16(_) => Dtype::BFloat16,
            Scalar::Float32(_) => Dtype::Float32,
            Scalar::Float64(_) => Dtype::Float64,
            Scalar::Int8(_) => Dtype::Int8,
            Scalar::Int16(_) => Dtype::Int16,
            Scalar::Int32(_) => Dtype::Int32,
            Scalar::Int64(_) => Dtype::Int64,
            Scalar::UInt8(_) => Dtype::UInt8,
            Scalar::UInt16(_) => Dtype::UInt16,
            Scalar::UInt32(_) => Dtype::UInt32,
            Scalar::UInt64(_) => Dtype::UInt64,
        }
    }

    #[inline(always)]
    pub fn store(&self, buf: &mut [u8]) {
        match self {
            Scalar::Boolean(x) => buf[0] = *x as u8,
            Scalar::Float16(x) => buf[..2].clone_from_slice(&x.to_le_bytes()),
            Scalar::BFloat16(x) => buf[..2].clone_from_slice(&x.to_le_bytes()),
            Scalar::Float32(x) => buf[..4].clone_from_slice(&x.to_le_bytes()),
            Scalar::Float64(x) => buf[..8].clone_from_slice(&x.to_le_bytes()),
            Scalar::Int8(x) => buf[0] = *x as u8,
            Scalar::Int16(x) => buf[..2].clone_from_slice(&x.to_le_bytes()),
            Scalar::Int32(x) => buf[..4].clone_from_slice(&x.to_le_bytes()),
            Scalar::Int64(x) => buf[..8].clone_from_slice(&x.to_le_bytes()),
            Scalar::UInt8(x) => buf[0] = *x,
            Scalar::UInt16(x) => buf[..2].clone_from_slice(&x.to_le_bytes()),
            Scalar::UInt32(x) => buf[..4].clone_from_slice(&x.to_le_bytes()),
            Scalar::UInt64(x) => buf[..8].clone_from_slice(&x.to_le_bytes()),
        }
    }
}

macro_rules! scalar_from {
    ($src:ty, $dst:tt, $as_fn:tt) => {
        impl From<Scalar> for $src {
            #[inline(always)]
            fn from(value: Scalar) -> $src {
                match value {
                    Scalar::$dst(x) => x,
                    _ => unreachable!(),
                }
            }
        }
        impl From<$src> for Scalar {
            #[inline(always)]
            fn from(value: $src) -> Self {
                Scalar::$dst(value)
            }
        }

        impl Scalar {
            #[inline(always)]
            pub fn $as_fn(&self) -> $src {
                match self {
                    Scalar::$dst(x) => *x,
                    _ => unreachable!(),
                }
            }
        }
    };
}

scalar_from!(bool, Boolean, as_bool);
scalar_from!(f16, Float16, as_f16);
scalar_from!(bf16, BFloat16, as_bf16);
scalar_from!(f32, Float32, as_f32);
scalar_from!(f64, Float64, as_f64);
scalar_from!(i8, Int8, as_i8);
scalar_from!(i16, Int16, as_i16);
scalar_from!(i32, Int32, as_i32);
scalar_from!(i64, Int64, as_i64);
scalar_from!(u8, UInt8, as_u8);
scalar_from!(u16, UInt16, as_u16);
scalar_from!(u32, UInt32, as_u32);
scalar_from!(u64, UInt64, as_u64);

impl From<usize> for Scalar {
    fn from(value: usize) -> Self {
        Self::UInt64(value as u64)
    }
}

impl From<isize> for Scalar {
    fn from(value: isize) -> Self {
        Self::Int64(value as i64)
    }
}

impl std::ops::Add<Self> for Scalar {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Scalar) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        match (self, rhs) {
            (Self::Boolean(_), Self::Boolean(_)) => unimplemented!("Can't add two booleans"),
            (Self::Float16(a), Self::Float16(b)) => (a + b).into(),
            (Self::BFloat16(a), Self::BFloat16(b)) => (a + b).into(),
            (Self::Float32(a), Self::Float32(b)) => (a + b).into(),
            (Self::Float64(a), Self::Float64(b)) => (a + b).into(),
            (Self::Int8(a), Self::Int8(b)) => (a + b).into(),
            (Self::Int16(a), Self::Int16(b)) => (a + b).into(),
            (Self::Int32(a), Self::Int32(b)) => (a + b).into(),
            (Self::Int64(a), Self::Int64(b)) => (a + b).into(),
            (Self::UInt8(a), Self::UInt8(b)) => (a + b).into(),
            (Self::UInt16(a), Self::UInt16(b)) => (a + b).into(),
            (Self::UInt32(a), Self::UInt32(b)) => (a + b).into(),
            (Self::UInt64(a), Self::UInt64(b)) => (a + b).into(),
            _ => unreachable!(),
        }
    }
}

impl std::ops::Sub<Self> for Scalar {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Scalar) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        match (self, rhs) {
            (Self::Boolean(_), Self::Boolean(_)) => unimplemented!("Can't sub two booleans"),
            (Self::Float16(a), Self::Float16(b)) => (a - b).into(),
            (Self::BFloat16(a), Self::BFloat16(b)) => (a - b).into(),
            (Self::Float32(a), Self::Float32(b)) => (a - b).into(),
            (Self::Float64(a), Self::Float64(b)) => (a - b).into(),
            (Self::Int8(a), Self::Int8(b)) => (a - b).into(),
            (Self::Int16(a), Self::Int16(b)) => (a - b).into(),
            (Self::Int32(a), Self::Int32(b)) => (a - b).into(),
            (Self::Int64(a), Self::Int64(b)) => (a - b).into(),
            (Self::UInt8(a), Self::UInt8(b)) => (a - b).into(),
            (Self::UInt16(a), Self::UInt16(b)) => (a - b).into(),
            (Self::UInt32(a), Self::UInt32(b)) => (a - b).into(),
            (Self::UInt64(a), Self::UInt64(b)) => (a - b).into(),
            _ => unreachable!(),
        }
    }
}

impl std::ops::Mul<Self> for Scalar {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Scalar) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        match (self, rhs) {
            (Self::Boolean(_), Self::Boolean(_)) => unimplemented!("Can't add two booleans"),
            (Self::Float16(a), Self::Float16(b)) => (a * b).into(),
            (Self::BFloat16(a), Self::BFloat16(b)) => (a * b).into(),
            (Self::Float32(a), Self::Float32(b)) => (a * b).into(),
            (Self::Float64(a), Self::Float64(b)) => (a * b).into(),
            (Self::Int8(a), Self::Int8(b)) => (a * b).into(),
            (Self::Int16(a), Self::Int16(b)) => (a * b).into(),
            (Self::Int32(a), Self::Int32(b)) => (a * b).into(),
            (Self::Int64(a), Self::Int64(b)) => (a * b).into(),
            (Self::UInt8(a), Self::UInt8(b)) => (a * b).into(),
            (Self::UInt16(a), Self::UInt16(b)) => (a * b).into(),
            (Self::UInt32(a), Self::UInt32(b)) => (a * b).into(),
            (Self::UInt64(a), Self::UInt64(b)) => (a * b).into(),
            _ => unreachable!(),
        }
    }
}

impl std::ops::Div<Self> for Scalar {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Scalar) -> Self::Output {
        assert_eq!(self.dtype(), rhs.dtype());
        match (self, rhs) {
            (Self::Boolean(_), Self::Boolean(_)) => unimplemented!("Can't add two booleans"),
            (Self::Float16(a), Self::Float16(b)) => (a / b).into(),
            (Self::BFloat16(a), Self::BFloat16(b)) => (a / b).into(),
            (Self::Float32(a), Self::Float32(b)) => (a / b).into(),
            (Self::Float64(a), Self::Float64(b)) => (a / b).into(),
            (Self::Int8(a), Self::Int8(b)) => (a / b).into(),
            (Self::Int16(a), Self::Int16(b)) => (a / b).into(),
            (Self::Int32(a), Self::Int32(b)) => (a / b).into(),
            (Self::Int64(a), Self::Int64(b)) => (a / b).into(),
            (Self::UInt8(a), Self::UInt8(b)) => (a / b).into(),
            (Self::UInt16(a), Self::UInt16(b)) => (a / b).into(),
            (Self::UInt32(a), Self::UInt32(b)) => (a / b).into(),
            (Self::UInt64(a), Self::UInt64(b)) => (a / b).into(),
            _ => unreachable!(),
        }
    }
}

impl Scalar {
    #[inline(always)]
    pub fn negate(self) -> Self {
        match self {
            Scalar::Boolean(a) => (!a).into(),
            Scalar::Float16(a) => (-a).into(),
            Scalar::BFloat16(a) => (-a).into(),
            Scalar::Float32(a) => (-a).into(),
            Scalar::Float64(a) => (-a).into(),
            Scalar::Int8(a) => (-a).into(),
            Scalar::Int16(a) => (-a).into(),
            Scalar::Int32(a) => (-a).into(),
            Scalar::Int64(a) => (-a).into(),
            _ => unimplemented!("Can't negate {self:?}"),
        }
    }
}

impl Scalar {
    #[inline(always)]
    pub fn recip(self) -> Self {
        match self {
            Scalar::Float16(a) => f16::from_f32(a.to_f32().recip()).into(),
            Scalar::BFloat16(a) => bf16::from_f32(a.to_f32().recip()).into(),
            Scalar::Float32(a) => a.recip().into(),
            Scalar::Float64(a) => a.recip().into(),
            _ => unimplemented!("Can't recip {self:?}"),
        }
    }
}

impl Scalar {
    #[inline(always)]
    pub fn to_dtype(self, dtype: Dtype) -> Self {
        if self.dtype() == dtype {
            return self;
        }

        match self {
            Scalar::Boolean(a) => match dtype {
                Dtype::Boolean => self,
                Dtype::Float16 => f16::from_f32(if a { 1.0 } else { 0.0 }).into(),
                Dtype::BFloat16 => bf16::from_f32(if a { 1.0 } else { 0.0 }).into(),
                Dtype::Float32 => if a { 1.0f32 } else { 0.0 }.into(),
                Dtype::Float64 => if a { 1.0f64 } else { 0.0 }.into(),
                Dtype::Int8 => if a { 1i8 } else { 0 }.into(),
                Dtype::Int16 => if a { 1i16 } else { 0 }.into(),
                Dtype::Int32 => if a { 1i32 } else { 0 }.into(),
                Dtype::Int64 => if a { 1i64 } else { 0 }.into(),
                Dtype::UInt8 => if a { 1u8 } else { 0 }.into(),
                Dtype::UInt16 => if a { 1u16 } else { 0 }.into(),
                Dtype::UInt32 => if a { 1u32 } else { 0 }.into(),
                Dtype::UInt64 => if a { 1u64 } else { 0 }.into(),
            },
            Scalar::Float16(a) => Scalar::Float32(a.to_f32()).to_dtype(dtype),
            Scalar::BFloat16(a) => Scalar::Float32(a.to_f32()).to_dtype(dtype),
            Scalar::Float32(a) => match dtype {
                Dtype::Boolean => (a != 0.0).into(),
                Dtype::Float16 => f16::from_f32(a).into(),
                Dtype::BFloat16 => bf16::from_f32(a).into(),
                Dtype::Float32 => self,
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::Float64(a) => match dtype {
                Dtype::Boolean => (a != 0.0).into(),
                Dtype::Float16 => f16::from_f64(a).into(),
                Dtype::BFloat16 => bf16::from_f64(a).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => self,
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::Int8(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => self,
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::Int16(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => self,
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::Int32(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => self,
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::Int64(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => self,
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::UInt8(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => self,
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::UInt16(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => self,
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::UInt32(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => self,
                Dtype::UInt64 => (a as u64).into(),
            },
            Scalar::UInt64(a) => match dtype {
                Dtype::Boolean => (a != 0).into(),
                Dtype::Float16 => f16::from_f64(a as f64).into(),
                Dtype::BFloat16 => bf16::from_f64(a as f64).into(),
                Dtype::Float32 => (a as f32).into(),
                Dtype::Float64 => (a as f64).into(),
                Dtype::Int8 => (a as i8).into(),
                Dtype::Int16 => (a as i16).into(),
                Dtype::Int32 => (a as i32).into(),
                Dtype::Int64 => (a as i64).into(),
                Dtype::UInt8 => (a as u8).into(),
                Dtype::UInt16 => (a as u16).into(),
                Dtype::UInt32 => (a as u32).into(),
                Dtype::UInt64 => self,
            },
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Axis(pub isize);

impl From<isize> for Axis {
    fn from(value: isize) -> Self {
        Self(value)
    }
}

impl Axis {
    pub fn get_value<T: Copy>(&self, dims: &[T]) -> T {
        dims[self.to_usize(dims.len())]
    }

    pub fn to_usize(&self, num_dims: usize) -> usize {
        self.0.rem_euclid(num_dims as isize) as usize
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
