use std::ops::Deref;

use cudarc::{driver::LaunchAsync, nvrtc::compile_ptx};

use crate::{init::build_tensor, tensor::*, util::*};

impl Tensor {
    pub fn mul(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
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
            assert!(self.deferred_dtype.num_bytes() <= self.stored_dtype.num_bytes());
            assert!(other.deferred_dtype.num_bytes() <= other.stored_dtype.num_bytes());

            let shape = &self.shape;

            let z_strides = crate::init::nd_bytes_strides(shape, dtype.num_bytes());

            let bytes = match (
                self.bytes_ptr.borrow().deref(),
                other.bytes_ptr.borrow().deref(),
            ) {
                (BytesPtr::Phantom, BytesPtr::Phantom) => BytesPtr::Phantom,
                (BytesPtr::Cpu(x_buf), BytesPtr::Cpu(y_buf)) => {
                    let x_prog = self.deferred_ops_cpu_closure();
                    let y_prog = other.deferred_ops_cpu_closure();

                    let mut z_buf = Vec::with_capacity(tensor_num_bytes);
                    z_buf.resize(tensor_num_bytes, 0);

                    let mut x_idx = CpuIndex::new(shape, &self.strides, self.byte_stride);
                    let mut y_idx = CpuIndex::new(shape, &other.strides, other.byte_stride);
                    for i_out in 0..numel {
                        let i_lhs = x_idx.next().unwrap();
                        let i_rhs = y_idx.next().unwrap();

                        let x_i = self.stored_dtype.read(&x_buf[i_lhs..]);
                        let y_i = other.stored_dtype.read(&y_buf[i_rhs..]);

                        let x_i = x_prog(&x_i);
                        let y_i = y_prog(&y_i);
                        let z_i = x_i * y_i;

                        z_i.store(&mut z_buf[i_out..]);
                    }
                    BytesPtr::Cpu(z_buf)
                }
                (BytesPtr::Cuda(x_buf), BytesPtr::Cuda(y_buf)) => {
                    let x_buf_ty = self.stored_dtype.cuda_type_name();
                    let y_buf_ty = other.stored_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();
                    let x_prog = self.deffered_ops_cuda_instructions();
                    let y_prog = other.deffered_ops_cuda_instructions();

                    let cuda = x_buf.device();

                    let mut z_buf = cuda.alloc_zeros::<u8>(num_bytes)?;

                    let module_name = std::format!(
                        "{}{}mul{}",
                        self.get_deferred_program_name(),
                        other.get_deferred_program_name(),
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

                    let mut info = Vec::new();
                    info.push(numel);
                    info.push(shape.len());
                    info.extend(shape);
                    info.extend(&self.strides);
                    info.extend(&other.strides);
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