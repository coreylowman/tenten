use std::{
    ops::{Deref, DerefMut},
    rc::Rc,
};

use cudarc::driver::LaunchAsync;

use crate::{tensor::*, util::*};

impl Tensor {
    pub fn add_assign(&mut self, other: Self) -> Result<(), Error> {
        let _no_grad = crate::backward::no_grad();

        if self.is_same_as(&other) {
            todo!()
        }

        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());
        assert_eq!(self.dtype(), other.dtype());

        let dtype = self.dtype();
        let numel = self.numel();

        assert!(self.deferred_dtype.num_bytes() <= self.byte_stride);

        let x_prog_name = self.build_op_name();
        let x_cpu_prog = self.build_cpu_op();
        let x_cuda_prog = self.build_cuda_op("x", "xf");
        self.deferred_ops.clear();

        let x_stored_dtype = self.stored_dtype;
        let x_byte_stride = self.byte_stride;
        let x_strides = self.strides.clone();
        let shape = &self.shape;

        match (
            Rc::make_mut(&mut self.bytes).borrow_mut().deref_mut(),
            other.bytes.borrow().deref(),
        ) {
            (BytesPtr::Ghost(_, _), _) => (),
            (BytesPtr::Cpu(x_buf), BytesPtr::Cpu(y_buf)) => {
                let x_prog = x_cpu_prog;
                let y_prog = other.build_cpu_op();

                let mut x_idx = CpuIndex::new(shape, &x_strides, x_byte_stride);
                let mut y_idx = CpuIndex::new(shape, &other.strides, other.byte_stride);
                for _ in 0..numel {
                    let i_lhs = x_idx.next().unwrap();
                    let i_rhs = y_idx.next().unwrap();

                    let x_i = x_stored_dtype.read(&x_buf[i_lhs..]);
                    let y_i = other.stored_dtype.read(&y_buf[i_rhs..]);

                    let x_i = x_prog(&x_i);
                    let y_i = y_prog(&y_i);
                    let z_i = x_i + y_i;

                    z_i.store(&mut x_buf[i_lhs..]);
                }
            }
            (BytesPtr::Cuda(x_buf), BytesPtr::Cuda(y_buf)) => {
                let x_buf_ty = x_stored_dtype.cuda_type_name();
                let y_buf_ty = other.stored_dtype.cuda_type_name();
                let dst_ty = dtype.cuda_type_name();
                let x_prog = x_cuda_prog;
                let y_prog = other.build_cuda_op("y", "yf");

                let cuda = y_buf.device();

                let module_name = std::format!(
                    "{}{}add_assign{}",
                    x_prog_name,
                    other.build_op_name(),
                    dtype.short_name()
                );

                if !cuda.has_func(&module_name, "kernel") {
                    let kernel_src = std::format!(
                        r#"
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
        for (int d = num_dims - 1; d >= 0; d--) {{
            size_t i_dim = tmp_i % dims[d];
            lhs_i += i_dim * lhs_strides[d];
            rhs_i += i_dim * rhs_strides[d];
            tmp_i /= dims[d];
        }}

        auto x = *reinterpret_cast<{x_buf_ty} *>(lhs + lhs_i);
        {x_prog}
        auto y = *reinterpret_cast<const {y_buf_ty} *>(rhs + rhs_i);
        {y_prog}
        *reinterpret_cast<{dst_ty} *>(lhs + lhs_i) = xf + yf;
    }}
}}
"#
                    );
                    let ptx = jit_compile(kernel_src)?;
                    cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                }

                let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                let mut info = Vec::new();
                info.push(numel);
                info.push(shape.len());
                info.extend(shape);
                info.extend(&x_strides);
                info.extend(&other.strides);
                let info = cuda.htod_copy(info)?;

                unsafe { fwd_fn.launch(launch_cfg::<128>(numel as u32), (&info, x_buf, y_buf)) }?;
            }
            _ => unreachable!(),
        };

        Ok(())
    }
}
