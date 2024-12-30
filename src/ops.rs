use std::{ops::Neg, rc::Rc};

use crate::{build_tensor, cuda::*, dtype::*, tensor::*};

use cudarc::{
    driver::{DeviceSlice, LaunchAsync},
    nvrtc::compile_ptx,
};
use half::{bf16, f16};

#[derive(Debug, Copy, Clone)]
pub struct Axis(pub isize);

impl From<isize> for Axis {
    fn from(value: isize) -> Self {
        Self(value)
    }
}

impl Axis {
    pub fn get<T: Copy>(&self, dims: &[T]) -> T {
        dims[self.to_usize(dims.len())]
    }

    pub fn to_usize(&self, num_dims: usize) -> usize {
        self.0.rem_euclid(num_dims as isize) as usize
    }
}

pub fn all_some<T, const N: usize>(arr: [Option<T>; N]) -> Option<[T; N]> {
    if arr.iter().all(Option::is_some) {
        Some(arr.map(Option::unwrap))
    } else {
        None
    }
}

impl Tensor {
    #[inline(always)]
    pub fn defer_op(mut self, name: &str, cpu_op: fn(&Scalar) -> Scalar, cuda_op: &str) -> Self {
        // TODO how will this work - most operations will always hang onto output for gradient, which will make this copy.
        // I guess that makes sense. but we will have tensor clones everywhere? Do we need another Rc<RefCell<>> around device ptr?
        let new_cell = Rc::make_mut(&mut self.0);
        {
            let mut data = new_cell.borrow_mut();
            data.deferred_ops.push(DeferredOp {
                name: name.into(),
                cpu_op: cpu_op.into(),
                cuda_op: cuda_op.into(),
            });
        }
        self
    }

    #[inline(always)]
    pub fn defer_op_with_args<Name: Into<String>, CudaOp: Into<String>>(
        mut self,
        name: Name,
        cpu_op: (fn(&Scalar, &[Scalar]) -> Scalar, Vec<Scalar>),
        cuda_op: CudaOp,
    ) -> Self {
        let new_cell = Rc::make_mut(&mut self.0);
        {
            let mut data = new_cell.borrow_mut();
            data.deferred_ops.push(DeferredOp {
                name: name.into(),
                cpu_op: cpu_op.into(),
                cuda_op: cuda_op.into(),
            });
        }
        self
    }

    pub fn undefer(self) -> Result<Self, Error> {
        if self.0.borrow().deferred_ops.len() == 0 {
            return Ok(self);
        }

        {
            let mut data = self.0.borrow_mut();

            let byte_stride = *data.strides.iter().filter(|&x| *x > 0).min().unwrap();
            let src_dtype = data.cur_dtype;
            let dst_dtype = data.deferred_dtype;

            let prog_name = data.get_deferred_program_name();
            let cpu_prog = data.deferred_ops_cpu_closure();
            let cuda_prog = data.deffered_ops_cuda_instructions();
            data.deferred_ops.clear();

            match &mut data.bytes {
                BytesPtr::Phantom => (),
                BytesPtr::Cpu(buf) => {
                    for i in (0..buf.len()).step_by(byte_stride) {
                        let x = src_dtype.read(&buf[i..]);
                        let y = cpu_prog(&x);
                        y.store(&mut buf[i..]);
                    }
                }
                BytesPtr::Cuda(buf) => {
                    let src_ty = src_dtype.cuda_type_name();
                    let dst_ty = dst_dtype.cuda_type_name();
                    let cuda = buf.device();

                    let module_name = std::format!("{prog_name}undefer");

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
#include "cuda_fp16.h"

extern "C" __global__ void kernel(const size_t *info, const uint8_t *buf) {{
    const size_t numel = info[0];
    const size_t byte_stride = info[1];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {{
        auto x = *static_cast<{src_ty} *>(buf + i * byte_stride);
        {cuda_prog}
        *static_cast<{dst_ty} *>(buf + i * byte_stride) = x;
    }}
}}
"#
                        );
                        let ptx = compile_ptx(kernel_src).unwrap();
                        cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                    }

                    let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                    let numel = buf.len() / byte_stride;

                    let mut info = Vec::with_capacity(2);
                    info.push(numel);
                    info.push(byte_stride);
                    let info = cuda.htod_copy(info)?;

                    unsafe { fwd_fn.launch(launch_cfg::<128>(numel as u32), (&info, buf)) }?;
                }
            }
        }

        Ok(self)
    }
}

impl TensorData {
    pub fn get_deferred_program_name(&self) -> String {
        self.deferred_ops
            .iter()
            .map(|op| op.name.clone())
            .collect::<Vec<String>>()
            .join("-")
    }

    #[inline]
    pub fn deferred_ops_cpu_closure(&self) -> impl Fn(&Scalar) -> Scalar {
        let ops: Vec<CpuOpPtr> = self
            .deferred_ops
            .iter()
            .map(|op| op.cpu_op.clone())
            .collect();
        move |a| {
            let mut x = *a;
            for op in ops.iter() {
                x = match op {
                    CpuOpPtr::Simple(f) => f(&x),
                    CpuOpPtr::WithOptions(f, args) => f(&x, args),
                };
            }
            x
        }
    }
    pub fn deffered_ops_cuda_instructions(&self) -> String {
        let mut prog = String::new();
        for op in self.deferred_ops.iter() {
            if op.cuda_op.contains("=") {
                // for when the type changes
                prog += &std::format!("{};\n", op.cuda_op);
            } else {
                prog += &std::format!("x = {};\n", op.cuda_op);
            }
        }
        prog
    }
}

mod cpu_utils {
    #[derive(Debug, Eq, PartialEq)]
    pub(crate) struct NdIndex<'a> {
        pub(crate) indices: Vec<usize>,
        pub(crate) shape: &'a [usize],
        pub(crate) strides: &'a [usize],
        pub(crate) next: Option<usize>,
        pub(crate) contiguous: Option<usize>,
    }

    impl<'a> NdIndex<'a> {
        #[inline]
        pub(crate) fn new(
            shape: &'a [usize],
            strides: &'a [usize],
            dtype: crate::dtype::Dtype,
        ) -> Self {
            Self {
                indices: Default::default(),
                shape,
                strides,
                next: if shape.len() > 0 { Some(0) } else { None },
                contiguous: (strides == &crate::init::nd_bytes_strides(shape, dtype))
                    .then(|| shape.iter().product::<usize>()),
            }
        }
    }

    impl<'a> NdIndex<'a> {
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
}

impl Tensor {
    pub fn fill_with_zeros(&self) -> Result<(), Error> {
        {
            let mut data = self.0.borrow_mut();
            match &mut data.bytes {
                BytesPtr::Phantom => (),
                BytesPtr::Cpu(vec) => vec.fill(0u8),
                BytesPtr::Cuda(cuda_slice) => {
                    let cuda = cuda_slice.device();
                    cuda.memset_zeros(cuda_slice)?;
                }
            }
            data.deferred_ops.clear();
        }
        if let Some(x_grad) = self.grad() {
            crate::record_op(move || x_grad.alloc()?.fill_with_zeros());
        }
        Ok(())
    }
}

impl Tensor {
    pub fn fill_with_ones(&self) -> Result<(), Error> {
        // TODO clear deferred ops
        // TODO add grad backward which sets grad to zero
        todo!()
    }
}

impl Tensor {
    pub fn copy(&self, other: Self) -> Result<(), Error> {
        if self.ptr_eq(&other) {
            return Ok(());
        }
        self.0.borrow_mut().clone_from(&other.0.borrow());
        Ok(())
    }
}

impl Tensor {
    pub fn add(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            todo!()
        }

        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        assert_eq!(self.dtype(), other.dtype());

        let dtype = self.dtype();
        let numel = self.numel();
        let tensor_num_bytes = numel * dtype.num_bytes();

        let z = {
            let x_data = self.0.borrow();
            let y_data = other.0.borrow();

            assert!(x_data.deferred_dtype.num_bytes() <= x_data.cur_dtype.num_bytes());
            assert!(y_data.deferred_dtype.num_bytes() <= y_data.cur_dtype.num_bytes());

            let shape = &x_data.shape;

            let z_strides = crate::init::nd_bytes_strides(shape, dtype);

            let bytes = match (&x_data.bytes, &y_data.bytes) {
                (BytesPtr::Phantom, BytesPtr::Phantom) => BytesPtr::Phantom,
                (BytesPtr::Cpu(x_buf), BytesPtr::Cpu(y_buf)) => {
                    let x_prog = x_data.deferred_ops_cpu_closure();
                    let y_prog = y_data.deferred_ops_cpu_closure();

                    let mut z_buf = Vec::with_capacity(tensor_num_bytes);
                    z_buf.fill(0);

                    let mut x_idx =
                        cpu_utils::NdIndex::new(shape, &x_data.strides, x_data.cur_dtype);
                    let mut y_idx =
                        cpu_utils::NdIndex::new(shape, &y_data.strides, y_data.cur_dtype);
                    for i_out in 0..numel {
                        let i_lhs = x_idx.next().unwrap();
                        let i_rhs = y_idx.next().unwrap();

                        let x_i = x_data.cur_dtype.read(&x_buf[i_lhs..]);
                        let y_i = y_data.cur_dtype.read(&y_buf[i_rhs..]);

                        let x_i = x_prog(&x_i);
                        let y_i = y_prog(&y_i);
                        let z_i = x_i + y_i;

                        z_i.store(&mut z_buf[i_out..]);
                    }
                    BytesPtr::Cpu(z_buf)
                }
                (BytesPtr::Cuda(x_buf), BytesPtr::Cuda(y_buf)) => {
                    let x_buf_ty = x_data.cur_dtype.cuda_type_name();
                    let y_buf_ty = y_data.cur_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();
                    let x_prog = x_data.deffered_ops_cuda_instructions();
                    let y_prog = y_data.deffered_ops_cuda_instructions();

                    let cuda = x_buf.device();

                    let mut z_buf = cuda.alloc_zeros::<u8>(tensor_num_bytes)?;

                    let module_name = std::format!(
                        "{}{}add{}",
                        x_data.get_deferred_program_name(),
                        y_data.get_deferred_program_name(),
                        dtype.short_name()
                    );

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
#include "cuda_fp16.h"

extern "C" __global__ void kernel(const size_t *info, const uint8_t *lhs, const uint8_t *rhs, uint8_t *out) {{
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t *dims = info + 2;
    const size_t *lhs_strides = info + 2 + num_dims;
    const size_t *rhs_strides = info + 2 + 2 * num_dims;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {{
        size_t tmp_i = i;
        size_t lhs_i = 0;
        size_t rhs_i = 0;
        for (size_t d = num_dims - 1; d >= 0; d--) {{
            size_t i_dim = tmp_i % dims[d];
            lhs_i += i_dim * lhs_strides[d];
            rhs_i += i_dim * rhs_strides[d];
            tmp_i /= dims[d];
        }}

        auto x = *static_cast<{x_buf_ty} *>(lhs + lhs_i);
        {x_prog}
        auto lhs = x;

        auto x = *static_Cast<{y_buf_ty} *>(rhs + rhs_i);
        {y_prog}
        auto rhs = x;

        *static_cast<{dst_ty} *>(out + i) = lhs + rhs;
    }}
}}
"#
                        );
                        let ptx = compile_ptx(kernel_src).unwrap();
                        cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                    }

                    let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                    let mut info = Vec::with_capacity(2 + 3 * shape.len());
                    info.push(numel);
                    info.push(shape.len());
                    info.extend(shape);
                    info.extend(&x_data.strides);
                    info.extend(&y_data.strides);
                    let info = cuda.htod_copy(info)?;

                    unsafe {
                        fwd_fn.launch(
                            launch_cfg::<128>(numel as u32),
                            (&info, x_buf, y_buf, &mut z_buf),
                        )
                    }?;

                    BytesPtr::Cuda(z_buf)
                }
                _ => unreachable!(),
            };

            build_tensor(
                dtype,
                shape.clone(),
                z_strides,
                bytes,
                self.requires_grad() && crate::backward::is_recording(),
            )
        };

        if let Some([x_grad, y_grad, z_grad]) = all_some([self.grad(), other.grad(), z.grad()]) {
            crate::backward::record_op(move || {
                x_grad.alloc()?.add_assign(z_grad.clone())?;
                y_grad.alloc()?.add_assign(z_grad)
            });
        }

        Ok(z)
    }
}

impl Tensor {
    /// NOTE: gradients not traced through this.
    pub fn add_assign(&self, other: Self) -> Result<(), Error> {
        let _no_grad = crate::backward::no_grad();

        if self.ptr_eq(&other) {
            todo!()
        }

        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        assert_eq!(self.dtype(), other.dtype());

        let dtype = self.dtype();
        let numel = self.numel();

        {
            let mut x_data = self.0.borrow_mut();
            let y_data = other.0.borrow();

            assert!(x_data.deferred_dtype.num_bytes() <= x_data.cur_dtype.num_bytes());

            let x_prog_name = x_data.get_deferred_program_name();
            let x_cpu_prog = x_data.deferred_ops_cpu_closure();
            let x_cuda_prog = x_data.deffered_ops_cuda_instructions();
            x_data.deferred_ops.clear();

            let x_storage_dtype = x_data.cur_dtype;
            let x_strides = x_data.strides.clone();

            let shape = &y_data.shape;

            match (&mut x_data.bytes, &y_data.bytes) {
                (BytesPtr::Phantom, BytesPtr::Phantom) => (),
                (BytesPtr::Cpu(x_buf), BytesPtr::Cpu(y_buf)) => {
                    let x_prog = x_cpu_prog;
                    let y_prog = y_data.deferred_ops_cpu_closure();

                    let mut x_idx = cpu_utils::NdIndex::new(shape, &x_strides, x_storage_dtype);
                    let mut y_idx =
                        cpu_utils::NdIndex::new(shape, &y_data.strides, y_data.cur_dtype);
                    for _ in 0..numel {
                        let i_lhs = x_idx.next().unwrap();
                        let i_rhs = y_idx.next().unwrap();

                        let x_i = x_storage_dtype.read(&x_buf[i_lhs..]);
                        let y_i = y_data.cur_dtype.read(&y_buf[i_rhs..]);

                        let x_i = x_prog(&x_i);
                        let y_i = y_prog(&y_i);
                        let z_i = x_i + y_i;

                        z_i.store(&mut x_buf[i_lhs..]);
                    }
                }
                (BytesPtr::Cuda(x_buf), BytesPtr::Cuda(y_buf)) => {
                    let x_buf_ty = x_storage_dtype.cuda_type_name();
                    let y_buf_ty = y_data.cur_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();
                    let x_prog = x_cuda_prog;
                    let y_prog = y_data.deffered_ops_cuda_instructions();

                    let cuda = y_buf.device();

                    let module_name = std::format!(
                        "{}{}add_assign{}",
                        x_prog_name,
                        y_data.get_deferred_program_name(),
                        dtype.short_name()
                    );

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
#include "cuda_fp16.h"

extern "C" __global__ void kernel(const size_t *info, const uint8_t *lhs, const uint8_t *rhs) {{
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t *dims = info + 2;
    const size_t *lhs_strides = info + 2 + num_dims;
    const size_t *rhs_strides = info + 2 + 2 * num_dims;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {{
        size_t tmp_i = i;
        size_t lhs_i = 0;
        size_t rhs_i = 0;
        for (size_t d = num_dims - 1; d >= 0; d--) {{
            size_t i_dim = tmp_i % dims[d];
            lhs_i += i_dim * lhs_strides[d];
            rhs_i += i_dim * rhs_strides[d];
            tmp_i /= dims[d];
        }}

        auto x = *static_cast<{x_buf_ty} *>(lhs + lhs_i);
        {x_prog}
        auto lhs = x;

        auto x = *static_cast<{y_buf_ty} *>(rhs + rhs_i);
        {y_prog}
        auto rhs = x;

        *static_cast<{dst_ty} *>(lhs + lhs_i) = lhs + rhs;
    }}
}}
"#
                        );
                        let ptx = compile_ptx(kernel_src).unwrap();
                        cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                    }

                    let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                    let mut info = Vec::with_capacity(2 + 3 * shape.len());
                    info.push(numel);
                    info.push(shape.len());
                    info.extend(shape);
                    info.extend(&x_strides);
                    info.extend(&y_data.strides);
                    let info = cuda.htod_copy(info)?;

                    unsafe {
                        fwd_fn.launch(launch_cfg::<128>(numel as u32), (&info, x_buf, y_buf))
                    }?;
                }
                _ => unreachable!(),
            };
        }

        Ok(())
    }
}

impl Tensor {
    pub fn add_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let x_ghost = self.clone();
        let y = self.defer_op_with_args(
            std::format!("add_{scalar:?}"),
            (|x, args| *x + args[0], vec![scalar]),
            std::format!("x + {scalar:?}"),
        );
        if let Some([x_grad, y_grad]) = all_some([x_ghost.grad(), y.grad()]) {
            crate::backward::record_op(move || x_grad.alloc()?.add_assign(y_grad.alloc()?));
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn sub(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            self.fill_with_zeros()?;
            Ok(self)
        } else {
            self.add(other.negate()?)
        }
    }
}

impl Tensor {
    pub fn sub_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        self.add_scalar(scalar.negate())
    }
}

impl Tensor {
    pub fn sub_assign(&self, other: Self) -> Result<(), Error> {
        let _no_grad = crate::backward::no_grad();
        if self.ptr_eq(&other) {
            self.fill_with_zeros()
        } else {
            self.add_assign(other.negate()?)
        }
    }
}

impl Tensor {
    pub fn mul(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return self.square();
        }

        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        assert_eq!(self.dtype(), other.dtype());

        let dtype = self.dtype();
        let numel = self.numel();
        let num_bytes = dtype.num_bytes();
        let tensor_num_bytes = numel * dtype.num_bytes();

        let z = {
            let x_data = self.0.borrow();
            let y_data = other.0.borrow();

            assert!(x_data.deferred_dtype.num_bytes() <= x_data.cur_dtype.num_bytes());
            assert!(y_data.deferred_dtype.num_bytes() <= y_data.cur_dtype.num_bytes());

            let shape = &x_data.shape;

            let z_strides = crate::init::nd_bytes_strides(shape, dtype);

            let bytes = match (&x_data.bytes, &y_data.bytes) {
                (BytesPtr::Phantom, BytesPtr::Phantom) => BytesPtr::Phantom,
                (BytesPtr::Cpu(x_buf), BytesPtr::Cpu(y_buf)) => {
                    let x_prog = x_data.deferred_ops_cpu_closure();
                    let y_prog = y_data.deferred_ops_cpu_closure();

                    let mut z_buf = Vec::with_capacity(tensor_num_bytes);
                    z_buf.fill(0);

                    let mut x_idx =
                        cpu_utils::NdIndex::new(shape, &x_data.strides, x_data.cur_dtype);
                    let mut y_idx =
                        cpu_utils::NdIndex::new(shape, &y_data.strides, x_data.cur_dtype);
                    for i_out in 0..numel {
                        let i_lhs = x_idx.next().unwrap();
                        let i_rhs = y_idx.next().unwrap();

                        let x_i = x_data.cur_dtype.read(&x_buf[i_lhs..]);
                        let y_i = y_data.cur_dtype.read(&y_buf[i_rhs..]);

                        let x_i = x_prog(&x_i);
                        let y_i = y_prog(&y_i);
                        let z_i = x_i * y_i;

                        z_i.store(&mut z_buf[i_out..]);
                    }
                    BytesPtr::Cpu(z_buf)
                }
                (BytesPtr::Cuda(x_buf), BytesPtr::Cuda(y_buf)) => {
                    let x_buf_ty = x_data.cur_dtype.cuda_type_name();
                    let y_buf_ty = y_data.cur_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();
                    let x_prog = x_data.deffered_ops_cuda_instructions();
                    let y_prog = y_data.deffered_ops_cuda_instructions();

                    let cuda = x_buf.device();

                    let mut z_buf = cuda.alloc_zeros::<u8>(num_bytes)?;

                    let module_name = std::format!(
                        "{}{}mul{}",
                        x_data.get_deferred_program_name(),
                        y_data.get_deferred_program_name(),
                        dtype.short_name()
                    );

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
#include "cuda_fp16.h"

extern "C" __global__ void kernel(const size_t *info, const uint8_t *lhs, const uint8_t *rhs, uint8_t *out) {{
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t *dims = info + 2;
    const size_t *lhs_strides = info + 2 + num_dims;
    const size_t *rhs_strides = info + 2 + 2 * num_dims;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {{
        size_t tmp_i = i;
        size_t lhs_i = 0;
        size_t rhs_i = 0;
        for (size_t d = num_dims - 1; d >= 0; d--) {{
            size_t i_dim = tmp_i % dims[d];
            lhs_i += i_dim * lhs_strides[d];
            rhs_i += i_dim * rhs_strides[d];
            tmp_i /= dims[d];
        }}

        auto x = *static_cast<{x_buf_ty} *>(lhs + lhs_i);
        {x_prog}
        auto lhs = x;

        auto x = *static_Cast<{y_buf_ty} *>(rhs + rhs_i);
        {y_prog}
        auto rhs = x;

        *static_cast<{dst_ty} *>(out + i) = lhs * rhs;
    }}
}}
"#
                        );
                        let ptx = compile_ptx(kernel_src).unwrap();
                        cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                    }

                    let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                    let mut info = Vec::with_capacity(2 + 3 * shape.len());
                    info.push(numel);
                    info.push(shape.len());
                    info.extend(shape);
                    info.extend(&x_data.strides);
                    info.extend(&y_data.strides);
                    let info = cuda.htod_copy(info)?;

                    unsafe {
                        fwd_fn.launch(
                            launch_cfg::<128>(numel as u32),
                            (&info, x_buf, y_buf, &mut z_buf),
                        )
                    }?;

                    BytesPtr::Cuda(z_buf)
                }
                _ => unreachable!(),
            };

            build_tensor(
                dtype,
                shape.clone(),
                z_strides,
                bytes,
                self.requires_grad() && crate::backward::is_recording(),
            )
        };

        if let Some([x_grad, y_grad, z_grad]) = all_some([self.grad(), other.grad(), z.grad()]) {
            crate::backward::record_op(move || {
                x_grad.alloc()?.add_assign(other.mul(z_grad.clone())?)?;
                y_grad.alloc()?.add_assign(self.mul(z_grad)?)
            });
        }

        Ok(z)
    }
}

impl Tensor {
    pub fn mul_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let x = self.clone();
        let y = self.defer_op_with_args(
            std::format!("mul_{scalar:?}"),
            (|x, args| *x * args[0], vec![scalar]),
            std::format!("x * {scalar:?}"),
        );
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                x_grad.alloc()?.add_assign(y_grad.mul_scalar(scalar)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn div(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            self.fill_with_ones()?;
            Ok(self)
        } else {
            // TODO handle integer values
            self.mul(other.recip()?)
        }
    }
}

impl Tensor {
    pub fn div_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        // TODO handle integer values
        self.mul_scalar(scalar.recip())
    }
}

impl Tensor {
    pub fn abs(self) -> Result<Self, Error> {
        let dtype = self.dtype();

        let x = self.clone();
        let y = match dtype {
            Dtype::Boolean => unimplemented!("Can't take abs of boolean tensor"),
            Dtype::UInt8 | Dtype::UInt16 | Dtype::UInt32 | Dtype::UInt64 => return Ok(self),
            Dtype::Float16 => self.defer_op(
                "absf16",
                |a| f16::from_f32(a.as_f16().to_f32().abs()).into(),
                "__habs(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "absbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().abs()).into(),
                "__habs(x)",
            ),
            Dtype::Float32 => self.defer_op("absf32", |a| a.as_f32().abs().into(), "fabsf(x)"),
            Dtype::Float64 => self.defer_op("absf64", |a| a.as_f64().abs().into(), "fabs(x)"),
            Dtype::Int8 => self.defer_op("absi8", |a| a.as_i8().abs().into(), "abs(x)"),
            Dtype::Int16 => self.defer_op("absi16", |a| a.as_i16().abs().into(), "abs(x)"),
            Dtype::Int32 => self.defer_op("absi32", |a| a.as_i32().abs().into(), "abs(x)"),
            Dtype::Int64 => self.defer_op("absi64", |a| a.as_i64().abs().into(), "abs(x)"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.sign()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn sign(self) -> Result<Self, Error> {
        // TODO can optimize this using replace code directly so we don't have to do 2x replace
        let dtype = self.dtype();
        let pos_mask = self.clone().gt_scalar(dtype.zero())?;
        let neg_mask = self.clone().lt_scalar(dtype.zero())?;
        self.replace(pos_mask, dtype.one())?
            .replace(neg_mask, dtype.one().negate())
    }
}

impl Tensor {
    pub fn sin(self) -> Result<Self, Error> {
        let dtype = self.dtype();

        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "sinf16",
                |a| f16::from_f32(a.as_f16().to_f32().sin()).into(),
                "hsin(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "sinbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().sin()).into(),
                "hsin(x)",
            ),
            Dtype::Float32 => self.defer_op("sinf32", |a| a.as_f32().sin().into(), "sinf(x)"),
            Dtype::Float64 => self.defer_op("sinf64", |a| a.as_f64().sin().into(), "sin(x)"),
            _ => unimplemented!("Can't take sin of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.cos()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn cos(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "cosf16",
                |a| f16::from_f32(a.as_f16().to_f32().cos()).into(),
                "hcos(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "cosbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().cos()).into(),
                "hcos(x)",
            ),
            Dtype::Float32 => self.defer_op("cosf32", |a| a.as_f32().cos().into(), "cosf(x)"),
            Dtype::Float64 => self.defer_op("cosf64", |a| a.as_f64().cos().into(), "cos(x)"),
            _ => unimplemented!("Can't take cos of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.sin()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn exp(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "expf16",
                |a| f16::from_f32(a.as_f16().to_f32().exp()).into(),
                "hexp(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "expbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().exp()).into(),
                "hexp(x)",
            ),
            Dtype::Float32 => self.defer_op("expf32", |a| a.as_f32().exp().into(), "expf(x)"),
            Dtype::Float64 => self.defer_op("expf64", |a| a.as_f64().exp().into(), "exp(x)"),
            _ => unimplemented!("Can't take exp of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.exp()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn ln(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "lnf16",
                |a| f16::from_f32(a.as_f16().to_f32().ln()).into(),
                "hlog(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "lnbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().ln()).into(),
                "hlog(x)",
            ),
            Dtype::Float32 => self.defer_op("lnf32", |a| a.as_f32().ln().into(), "logf(x)"),
            Dtype::Float64 => self.defer_op("lnf64", |a| a.as_f64().ln().into(), "log(x)"),
            _ => unimplemented!("Can't take ln of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.recip()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn recip(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "recipf16",
                |a| f16::from_f32(a.as_f16().to_f32().recip()).into(),
                "__half(1.0) / x",
            ),
            Dtype::BFloat16 => self.defer_op(
                "recipbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().recip()).into(),
                "__nv_bfloat16(1.0) / x",
            ),
            Dtype::Float32 => self.defer_op("recipf32", |a| a.as_f32().recip().into(), "1.0 / x"),
            Dtype::Float64 => self.defer_op("recipf64", |a| a.as_f64().recip().into(), "1.0 / x"),
            _ => unimplemented!("Can't take recip of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.recip()?.square()?.negate()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn square(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "squaref16",
                |a| f16::from_f32(a.as_f16().to_f32().powi(2)).into(),
                "x*x",
            ),
            Dtype::BFloat16 => self.defer_op(
                "squarebf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().powi(2)).into(),
                "x*x",
            ),
            Dtype::Float32 => self.defer_op("squaref32", |a| a.as_f32().powi(2).into(), "x*x"),
            Dtype::Float64 => self.defer_op("squaref64", |a| a.as_f64().powi(2).into(), "x*x"),
            Dtype::Int8 => self.defer_op("squarei8", |a| a.as_i8().pow(2).into(), "x*x"),
            Dtype::Int16 => self.defer_op("squarei16", |a| a.as_i16().pow(2).into(), "x*x"),
            Dtype::Int32 => self.defer_op("squarei32", |a| a.as_i32().pow(2).into(), "x*x"),
            Dtype::Int64 => self.defer_op("squarei64", |a| a.as_i64().pow(2).into(), "x*x"),
            Dtype::UInt8 => self.defer_op("squareu8", |a| a.as_u8().pow(2).into(), "x*x"),
            Dtype::UInt16 => self.defer_op("squareu16", |a| a.as_u16().pow(2).into(), "x*x"),
            Dtype::UInt32 => self.defer_op("squareu32", |a| a.as_u32().pow(2).into(), "x*x"),
            Dtype::UInt64 => self.defer_op("squareu64", |a| a.as_u64().pow(2).into(), "x*x"),
            _ => unimplemented!("Can't take square of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.mul_scalar(2.0f64)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn sqrt(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "sqrtf16",
                |a| f16::from_f32(a.as_f16().to_f32().sqrt()).into(),
                "hsqrt(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "sqrtbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().sqrt()).into(),
                "hsqrt(x)",
            ),
            Dtype::Float32 => self.defer_op("sqrtf32", |a| a.as_f32().sqrt().into(), "sqrtf(x)"),
            Dtype::Float64 => self.defer_op("sqrtf64", |a| a.as_f64().sqrt().into(), "sqrt(x)"),
            _ => unimplemented!("Can't take sqrt of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.sqrt()?.mul_scalar(2.0f64)?.recip()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn negate(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "negatef16",
                |a| f16::from_f32(a.as_f16().to_f32().neg()).into(),
                "-x",
            ),
            Dtype::BFloat16 => self.defer_op(
                "negatebf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().neg()).into(),
                "-x",
            ),
            Dtype::Float32 => self.defer_op("negatef32", |a| a.as_f32().neg().into(), "-x"),
            Dtype::Float64 => self.defer_op("negatef64", |a| a.as_f64().neg().into(), "-x"),
            Dtype::Int8 => self.defer_op("negatei8", |a| a.as_i8().neg().into(), "-x"),
            Dtype::Int16 => self.defer_op("negatei16", |a| a.as_i16().neg().into(), "-x"),
            Dtype::Int32 => self.defer_op("negatei32", |a| a.as_i32().neg().into(), "-x"),
            Dtype::Int64 => self.defer_op("negatei64", |a| a.as_i64().neg().into(), "-x"),
            _ => unimplemented!("Can't take negate of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || x_grad.alloc()?.add_assign(y_grad.negate()?));
        }
        Ok(y)
    }
}

impl Tensor {
    /// TODO feature gate this
    pub fn gelu_true(self) -> Result<Self, Error> {
        todo!("defer_op")
    }
}

impl Tensor {
    pub fn gelu_approx(self) -> Result<Self, Error> {
        todo!("defer_op")
    }
}

impl Tensor {
    pub fn clamp<S: Into<Scalar>>(self, min: S, max: S) -> Result<Self, Error> {
        self.max_scalar(min)?.min_scalar(max)
    }
}

impl Tensor {
    pub fn pow(self, exponent: f64) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op_with_args(
                std::format!("powf16_{exponent:?}"),
                (
                    |a, args| f16::from_f32(a.as_f16().to_f32().powf(args[0].as_f32())).into(),
                    vec![Scalar::Float32(exponent as f32)],
                ),
                std::format!("__float2half(powf(__half2float(x), {exponent:?}))"),
            ),
            Dtype::BFloat16 => self.defer_op_with_args(
                std::format!("powbf16_{exponent:?}"),
                (
                    |a, args| bf16::from_f32(a.as_bf16().to_f32().powf(args[0].as_f32())).into(),
                    vec![Scalar::Float32(exponent as f32)],
                ),
                std::format!("__float2half(powf(__half2float(x), {exponent:?}))"),
            ),
            Dtype::Float32 => self.defer_op_with_args(
                std::format!("powf32_{exponent:?}"),
                (
                    |a, args| a.as_f32().powf(args[1].as_f32()).into(),
                    vec![Scalar::Float32(exponent as f32)],
                ),
                std::format!("powf(x, {exponent:?})"),
            ),
            Dtype::Float64 => self.defer_op_with_args(
                std::format!("powf64_{exponent:?}"),
                (
                    |a, args| a.as_f64().powf(args[1].as_f64()).into(),
                    vec![Scalar::Float64(exponent)],
                ),
                std::format!("pow(x, {exponent:?})"),
            ),
            _ => unimplemented!("Can't take pow of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.pow(exponent - 1.0)?.mul_scalar(exponent)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn relu(self) -> Result<Self, Error> {
        let zero = self.dtype().zero();
        self.max_scalar(zero)
    }
}

impl Tensor {
    pub fn sigmoid(self) -> Result<Self, Error> {
        fn sigmoidf(x: f32) -> f32 {
            1.0 / (1.0 + x.neg().exp())
        }

        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "sigmoidf16",
                |a| f16::from_f32(sigmoidf(a.as_f16().to_f32())).into(),
                "__half(1.0) / (__half(1.0) + hexp(-x))",
            ),
            Dtype::BFloat16 => self.defer_op(
                "sigmoidbf16",
                |a| bf16::from_f32(sigmoidf(a.as_bf16().to_f32())).into(),
                "__nv_bfloat16(1.0) / (__nv_bfloat16(1.0) + hexp(-x))",
            ),
            Dtype::Float32 => self.defer_op(
                "sigmoidf32",
                |a| sigmoidf(a.as_f32()).into(),
                "1.0 / (1.0 + expf(-x))",
            ),
            Dtype::Float64 => self.defer_op(
                "sigmoidf64",
                |a| (1.0 / (1.0 + a.as_f64().neg().exp())).into(),
                "1.0 / (1.0 + exp(-x))",
            ),
            _ => unimplemented!("Can't take sigmoid of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let y = x.sigmoid()?;
                let dfdx = y.clone().mul(y.negate()?.add_scalar(dtype.one())?)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn tanh(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "tanhf16",
                |a| f16::from_f32(a.as_f16().to_f32().tanh()).into(),
                "__float2half(tanhf(__half2float(a)))",
            ),
            Dtype::BFloat16 => self.defer_op(
                "tanhbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().tanh()).into(),
                "__float2half(tanhf(__half2float(a)))",
            ),
            Dtype::Float32 => self.defer_op("tanhf32", |a| a.as_f32().tanh().into(), "tanhf(x)"),
            Dtype::Float64 => self.defer_op("tanhf64", |a| a.as_f64().tanh().into(), "tanh(x)"),
            _ => unimplemented!("Can't take tanh of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.tanh()?.square()?.negate()?.add_scalar(dtype.one())?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn to_dtype(self, dst: Dtype) -> Result<Self, Error> {
        let src = self.dtype();
        if src == dst {
            Ok(self)
        } else if dst.num_bytes() <= src.num_bytes() {
            let x = self.clone();
            let y: Tensor = self.defer_op_with_args(
                std::format!("to_{}", dst.short_name()),
                (|x, args| x.to_dtype(args[0].dtype()), vec![dst.zero()]),
                std::format!("{} x = x", dst.cuda_type_name()),
            );
            {
                let mut y_data = y.0.borrow_mut();
                y_data.deferred_dtype = dst;
            }
            if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
                crate::backward::record_op(move || {
                    x_grad.alloc()?.add_assign(y_grad.to_dtype(src)?)
                });
            }
            Ok(y)
        } else {
            todo!("can't defer this, need to allocate more or less space")
        }
    }
}

impl Tensor {
    pub fn to_device(self, device: Device) -> Result<Self, Error> {
        // TODO can keep deferred ops and just send existing data to the device
        todo!()
    }
}

impl Tensor {
    pub fn roll_along<A: Into<Axis>>(self, axis: A, shift: isize) -> Result<Self, Error> {
        todo!()
    }
}

pub trait Stack {
    fn stack(self) -> Result<Tensor, Error>;
}

impl<Tensors: Into<Vec<Tensor>>> Stack for Tensors {
    fn stack(self) -> Result<Tensor, Error> {
        let tensors = Into::<Vec<Tensor>>::into(self);
        todo!()
    }
}

pub trait ConcatAlong {
    fn concat_along<A: Into<Axis>>(self, axis: A) -> Result<Tensor, Error>;
}

impl<Tensors: Into<Vec<Tensor>>> ConcatAlong for Tensors {
    fn concat_along<A: Into<Axis>>(self, axis: A) -> Result<Tensor, Error> {
        let tensors = Into::<Vec<Tensor>>::into(self);
        todo!()
    }
}

impl Tensor {
    pub fn broadcast_along<A: Into<Axis>>(mut self, axis: A, size: usize) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let new_cell = Rc::make_mut(&mut self.0);
        {
            let mut data = new_cell.borrow_mut();
            let dim = axis.to_usize(data.shape.len() + 1);
            data.shape.insert(dim, size);
            data.strides.insert(dim, 0);
        }
        Ok(self)
    }

    pub fn broadcast_like<A: Into<Axis>>(mut self, axis: A, other: &Self) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let tgt_shape = other.shape();
        let new_cell = Rc::make_mut(&mut self.0);
        {
            let mut data = new_cell.borrow_mut();
            let dim = axis.to_usize(data.shape.len() + 1);
            data.shape.insert(dim, tgt_shape[dim]);
            data.strides.insert(dim, 0);
            assert_eq!(
                data.shape, tgt_shape,
                "After broadcasting {axis:?}, {:?} does not match {tgt_shape:?}.",
                data.shape
            );
        }
        Ok(self)
    }
}

impl Tensor {
    pub fn permute(mut self, order: &[isize]) -> Result<Self, Error> {
        let num_dims = self.shape().len();

        assert_eq!(num_dims, order.len());
        let mut dup_found = false;
        for i in 0..num_dims {
            for j in i + 1..num_dims {
                if order[i] == order[j] {
                    dup_found = true;
                }
            }
        }
        assert!(
            !dup_found,
            "Must specify each dimension exactly once in permute command"
        );

        let new_cell = Rc::make_mut(&mut self.0);
        {
            let mut data = new_cell.borrow_mut();
            let mut new_shape = data.shape.clone();
            let mut new_strides = data.strides.clone();
            for (i, new_dim) in order.iter().enumerate() {
                let new_dim = new_dim.rem_euclid(num_dims as isize) as usize;
                new_strides[i] = new_dim;
                new_shape[i] = data.shape[new_dim];
            }
            data.shape.clone_from(&new_shape);
            data.strides.clone_from(&new_strides);
        }

        Ok(self)
    }
}

impl Tensor {
    pub fn reshape_like(self, other: &Self) -> Result<Self, Error> {
        self.reshape(other.shape())
    }

    pub fn contiguous(self) -> Result<Self, Error> {
        let shape = self.shape();
        self.reshape(shape)
    }

    pub fn reshape<Shape: Into<Vec<usize>>>(self, shape: Shape) -> Result<Self, Error> {
        todo!()
    }
}

impl Tensor {
    pub fn sum_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        // A number of cases to handle:
        // 1. Contiguous
        // 2. Permuted
        // 2. Axis is broadcasted
        let axis = Into::<Axis>::into(axis);
        let x = self.clone();
        let y: Tensor = todo!("forward");
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let y_grad = y_grad.broadcast_like(axis, &x_grad)?;
                x_grad.alloc()?.add_assign(y_grad)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn sum(mut self) -> Result<Self, Error> {
        let shape = self.shape();
        for _ in 0..shape.len() {
            self = self.sum_along(-1)?;
        }
        assert_eq!(self.shape(), vec![]);
        Ok(self)
    }
}

impl Tensor {
    pub fn mean_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let shape = self.shape();
        self.sum_along(axis)?
            .mul_scalar(1.0f64 / (axis.get(&shape) as f64))
    }
}

impl Tensor {
    pub fn mean(self) -> Result<Self, Error> {
        let shape = self.shape();
        let num_elem = shape.iter().product::<usize>();
        self.sum()?.mul_scalar(1.0f64 / (num_elem as f64))
    }
}

impl Tensor {
    pub fn var_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();
        self.clone()
            .mean_along(axis)?
            .broadcast_like(axis, &self)?
            .sub(self)?
            .square()?
            .mean_along(axis)
    }
}

impl Tensor {
    pub fn std_along<A: Into<Axis>, S: Into<Scalar>>(self, axis: A, eps: S) -> Result<Self, Error> {
        self.var_along(axis)?.add_scalar(eps)?.sqrt()
    }
}

impl Tensor {
    pub fn min(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            Ok(self)
        } else {
            // TODO we can optimize this with a special kernel
            self.clone().le(other.clone())?.choose(self, other)
        }
    }
}

impl Tensor {
    pub fn min_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        todo!()
    }
}

impl Tensor {
    pub fn min_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let dtype = self.dtype();
        let x = self.clone();
        let y = self.defer_op_with_args(
            std::format!("min_{scalar:?}"),
            (
                |a, args| {
                    if *a < args[0] {
                        args[0]
                    } else {
                        *a
                    }
                },
                vec![scalar],
            ),
            std::format!("(x < {scalar:?} ? {scalar:?} : x)"),
        );
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.lt_scalar(scalar)?.to_dtype(dtype)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn max(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            Ok(self)
        } else {
            // TODO we can optimize this with a special kernel
            self.clone().ge(other.clone())?.choose(self, other)
        }
    }
}

impl Tensor {
    pub fn max_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        todo!()
    }
}

impl Tensor {
    pub fn max_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let dtype = self.dtype();
        let x = self.clone();
        let y = self.defer_op_with_args(
            std::format!("max_{scalar:?}"),
            (
                |a, args| {
                    if *a > args[0] {
                        args[0]
                    } else {
                        *a
                    }
                },
                vec![scalar],
            ),
            std::format!("(x > {scalar:?} ? {scalar:?} : x)"),
        );
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.gt_scalar(scalar)?.to_dtype(dtype)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn logsumexp_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();

        let no_grad = crate::backward::no_grad();
        let max = self.clone().max_along(axis)?;
        self.sub_assign(max.clone().broadcast_like(axis, &self)?)?;
        drop(no_grad);

        let x = self.exp()?.sum_along(axis)?.ln()?;

        let no_grad = crate::backward::no_grad();
        x.add_assign(max)?;
        drop(no_grad);

        Ok(x)
    }
}

impl Tensor {
    pub fn log_softmax_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();

        let no_grad = crate::backward::no_grad();
        let max = self.clone().max_along(axis)?.broadcast_like(axis, &self)?;
        self.sub_assign(max)?;
        drop(no_grad);

        let logsumexp = self
            .clone()
            .exp()?
            .sum_along(axis)?
            .ln()?
            .broadcast_like(axis, &self)?;
        self.sub(logsumexp)
    }
}

impl Tensor {
    pub fn softmax_along<A: Into<Axis>>(self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();

        let no_grad = crate::backward::no_grad();
        let max = self.clone().max_along(axis)?.broadcast_like(axis, &self)?;
        self.sub_assign(max)?;
        drop(no_grad);

        let x_exp = self.clone().exp()?;
        let x_expsum = x_exp
            .clone()
            .sum_along(axis)?
            .broadcast_like(axis, &x_exp)?;
        x_exp.div(x_expsum)
    }
}

impl Tensor {
    pub fn dot(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            todo!()
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.device(), other.device());

        let lhs_shape = self.shape();
        let rhs_shape = other.shape();

        // dim checks
        match (&lhs_shape[..], &rhs_shape[..]) {
            ([m], [n]) => (),
            ([k1], [k2, n]) => assert_eq!(k1, k2),
            ([m, k1], [k2]) => assert_eq!(k1, k2),
            ([m, k1], [k2, n]) => assert_eq!(k1, k2),
            ([b, m, k1], [k2, n]) => assert_eq!(k1, k2),
            ([b1, m, k1], [b2, k2, n]) => {
                assert_eq!(b1, b2);
                assert_eq!(k1, k2);
            }
            ([b1, s1, m, k1], [b2, s2, k2, n]) => {
                assert_eq!(b1, b2);
                assert_eq!(s1, s2);
                assert_eq!(k1, k2);
            }
            (a, b) => unimplemented!("Unable to run dot product on shapes {a:?}x{b:?}"),
        };
        todo!()
    }
}

impl Tensor {
    pub fn eq(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return crate::init::full(self.shape(), true)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("mutate dtype")
    }
}

impl Tensor {
    pub fn eq_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let dtype = self.dtype();
        let scalar = Into::<Scalar>::into(scalar).to_dtype(dtype);
        let y = self.defer_op_with_args(
            std::format!("eq_{scalar:?}"),
            (|x, args| Scalar::Boolean(*x == args[1]), vec![scalar]),
            std::format!("x == {scalar:?}"),
        );
        {
            let mut y_data = y.0.borrow_mut();
            y_data.deferred_dtype = Dtype::Boolean;
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn ne(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return crate::init::full(self.shape(), false)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!()
    }
}

impl Tensor {
    pub fn ne_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let y = self.defer_op_with_args(
            std::format!("ne_{scalar:?}"),
            (|x, args| Scalar::Boolean(*x != args[1]), vec![scalar]),
            std::format!("x != {scalar:?}"),
        );
        {
            let mut y_data = y.0.borrow_mut();
            y_data.deferred_dtype = Dtype::Boolean;
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn gt(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return crate::init::full(self.shape(), false)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("change dtype");
        todo!()
    }
}

impl Tensor {
    pub fn gt_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let y = self.defer_op_with_args(
            std::format!("gt_{scalar:?}"),
            (|x, args| Scalar::Boolean(*x > args[1]), vec![scalar]),
            std::format!("x > {scalar:?}"),
        );
        {
            let mut y_data = y.0.borrow_mut();
            y_data.deferred_dtype = Dtype::Boolean;
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn ge(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return crate::init::full(self.shape(), true)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("change dtype");
        todo!()
    }
}

impl Tensor {
    pub fn ge_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let y = self.defer_op_with_args(
            std::format!("ge_{scalar:?}"),
            (|x, args| Scalar::Boolean(*x >= args[1]), vec![scalar]),
            std::format!("x >= {scalar:?}"),
        );
        {
            let mut y_data = y.0.borrow_mut();
            y_data.deferred_dtype = Dtype::Boolean;
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn lt(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return crate::init::full(self.shape(), false)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("change dtype");
        todo!()
    }
}

impl Tensor {
    pub fn lt_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let y = self.defer_op_with_args(
            std::format!("lt_scalar_{scalar:?}"),
            (|x, args| Scalar::Boolean(*x < args[1]), vec![scalar]),
            std::format!("x < {scalar:?}"),
        );
        {
            let mut y_data = y.0.borrow_mut();
            y_data.deferred_dtype = Dtype::Boolean;
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn le(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return crate::init::full(self.shape(), true)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("change dtype");
        todo!()
    }
}

impl Tensor {
    pub fn le_scalar<S: Into<Scalar>>(self, scalar: S) -> Result<Self, Error> {
        let scalar = Into::<Scalar>::into(scalar).to_dtype(self.dtype());
        let y = self.defer_op_with_args(
            std::format!("le_scalar_{scalar:?}"),
            (|x, args| Scalar::Boolean(*x <= args[1]), vec![scalar]),
            std::format!("x <= {scalar:?}"),
        );
        {
            let mut y_data = y.0.borrow_mut();
            y_data.deferred_dtype = Dtype::Boolean;
        }
        Ok(y)
    }
}

impl Tensor {
    pub fn not(self) -> Result<Self, Error> {
        assert_eq!(self.dtype(), Dtype::Boolean);
        todo!("defer_op, backwards not supported")
    }
}

impl Tensor {
    pub fn and(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return Ok(self);
        }

        assert_eq!(self.dtype(), Dtype::Boolean);
        assert_eq!(other.dtype(), Dtype::Boolean);
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("defer_op, backwards not supported")
    }
}

impl Tensor {
    pub fn or(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            return Ok(self);
        }

        assert_eq!(self.dtype(), Dtype::Boolean);
        assert_eq!(other.dtype(), Dtype::Boolean);
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("defer_op, backwards not supported")
    }
}

impl Tensor {
    pub fn xor(self, other: Self) -> Result<Self, Error> {
        if self.ptr_eq(&other) {
            self.fill_with_zeros()?;
            return Ok(self);
        }

        assert_eq!(self.dtype(), Dtype::Boolean);
        assert_eq!(other.dtype(), Dtype::Boolean);
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        todo!("defer_op, backwards not supported")
    }
}

impl Tensor {
    pub fn dropout(self, prob: f64) -> Result<Self, Error> {
        let mask = self.sample_uniform_like().le_scalar(prob)?;
        self.div_scalar(1.0 - prob)?.replace(mask, 0.0f64)
    }
}

impl Tensor {
    pub fn gather_along<A: Into<Axis>>(self, axis: A, indices: Self) -> Result<Self, Error> {
        todo!()
    }
}

impl Tensor {
    pub fn choose(self, a: Self, b: Self) -> Result<Self, Error> {
        if a.ptr_eq(&b) {
            return Ok(a);
        }
        assert_eq!(self.dtype(), Dtype::Boolean);
        assert_eq!(a.dtype(), b.dtype());
        assert_eq!(self.shape(), a.shape());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(self.device(), a.device());
        assert_eq!(a.device(), b.device());
        todo!()
    }
}

impl Tensor {
    pub fn replace<S: Into<Scalar>>(self, cond: Self, value: S) -> Result<Self, Error> {
        assert_eq!(cond.dtype(), Dtype::Boolean);
        assert_eq!(self.shape(), cond.shape());
        assert_eq!(self.device(), cond.device());

        let scalar = Into::<Scalar>::into(value).to_dtype(self.dtype());
        todo!()
    }
}

impl Tensor {
    pub fn slice_along<A: Into<Axis>, R: std::ops::RangeBounds<usize>>(
        self,
        axis: A,
        range: R,
    ) -> Result<Self, Error> {
        todo!()
    }
}

impl Tensor {
    pub fn mse_loss(self, target: Self) -> Result<Self, Error> {
        self.sub(target)?.square()?.mean()
    }
}

impl Tensor {
    pub fn rmse_loss(self, target: Self) -> Result<Self, Error> {
        self.mse_loss(target)?.sqrt()
    }
}

impl Tensor {
    pub fn mae_loss(self, target: Self) -> Result<Self, Error> {
        self.sub(target)?.abs()?.mean()
    }
}

impl Tensor {
    pub fn huber_loss<S: Into<Scalar>>(self, target: Self, delta: S) -> Result<Self, Error> {
        let dtype = self.dtype();
        let delta = Into::<Scalar>::into(delta).to_dtype(dtype);
        let diff = self.sub(target)?;
        let a = diff.clone().square()?.mul_scalar(0.5f64)?;
        let b = diff
            .clone()
            .abs()?
            .sub_scalar(delta / Into::<Scalar>::into(2.0f64).to_dtype(dtype))?
            .mul_scalar(delta)?;
        diff.lt_scalar(delta)?.choose(a, b)?.mean()
    }
}

impl Tensor {
    pub fn smooth_l1_loss<S: Into<Scalar>>(self, target: Self, delta: S) -> Result<Self, Error> {
        let delta = Into::<Scalar>::into(delta).to_dtype(self.dtype());
        self.huber_loss(target, delta)?.div_scalar(delta)
    }
}

impl Tensor {
    pub fn xent_with_logits_loss(self, target_probs: Self) -> Result<Self, Error> {
        assert_eq!(self.shape(), target_probs.shape());
        assert_eq!(self.dtype(), target_probs.dtype());

        let shape = self.shape();
        self.log_softmax_along(-1)?
            .mul(target_probs)?
            .mean()?
            .negate()?
            .mul_scalar(*shape.last().unwrap())
    }
}

impl Tensor {
    pub fn kldiv_with_logits_loss(self, target_probs: Self) -> Result<Self, Error> {
        assert_eq!(self.shape(), target_probs.shape());
        assert_eq!(self.dtype(), target_probs.dtype());

        let shape = self.shape();
        self.log_softmax_along(-1)?
            .sub(target_probs.clone().ln()?)?
            .mul(target_probs)?
            .mean()?
            .negate()?
            .mul_scalar(*shape.last().unwrap())
    }
}

impl Tensor {
    pub fn binary_xent_with_logits_loss(self, target_probs: Self) -> Result<Self, Error> {
        assert_eq!(self.dtype(), target_probs.dtype());
        assert_eq!(self.shape(), target_probs.shape());

        let dtype = self.dtype();
        let a = self.clone().max_scalar(dtype.zero())?; // NOTE: fused
        let b = self.clone().mul(target_probs)?; // NOTE: eager
        let c = self.abs()?.negate()?.exp()?.add_scalar(dtype.one())?.ln()?; // NOTE: fused
        a.sub(b)?.add(c)?.mean()
    }
}
