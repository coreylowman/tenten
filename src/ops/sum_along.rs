use std::ops::Deref;

use cudarc::{driver::LaunchAsync, nvrtc::compile_ptx};

use crate::{init::build_tensor, tensor::*, util::*};

impl Tensor {
    pub fn sum_along<A: Into<Axis>>(mut self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let ax = axis.to_usize(self.shape.len());
        let x = self.clone();
        let y = if self.shape[ax] == 1 {
            self.shape.remove(ax);
            self.strides.remove(ax);
            self
        } else if self.strides[ax] == 0 {
            let scale = self.shape.remove(ax);
            self.strides.remove(ax);
            // TODO does this work properly with deferred ops?
            self.mul_scalar(scale)?
        } else {
            let dtype = self.deferred_dtype;
            let mut shape = self.shape.clone();
            shape.remove(ax);
            let strides = crate::init::nd_bytes_strides(&shape, dtype.num_bytes());
            let y_numel: usize = shape.iter().product();
            let y_num_bytes = y_numel * dtype.num_bytes();
            let reduced_dim = self.shape[ax];

            let bytes = match self.bytes.borrow().deref() {
                BytesPtr::Cpu(x_buf) => {
                    let prog = self.deferred_ops_cpu_closure();

                    let mut y_buf = vec![0; y_num_bytes];

                    let mut idx = CpuIndex::new(&self.shape, &self.strides, self.byte_stride);

                    for i_y in (0..y_num_bytes).step_by(dtype.num_bytes()) {
                        // TODO do accumulation in f32 for f16/bf16
                        let mut y = dtype.zero();
                        for _ in 0..reduced_dim {
                            let i_x = idx.next().unwrap();
                            let x = self.stored_dtype.read(&x_buf[i_x..]);
                            y = y + prog(&x);
                        }
                        y.store(&mut y_buf[i_y..]);
                    }

                    assert!(idx.next().is_none());

                    BytesPtr::Cpu(y_buf)
                }
                BytesPtr::Cuda(x_buf) => {
                    let cuda = x_buf.device();

                    let prog = self.deffered_ops_cuda_instructions();
                    let x_buf_ty = self.stored_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();

                    let mut y_buf = cuda.alloc_zeros(y_num_bytes)?;

                    let module_name = std::format!(
                        "{}sum{ax:?}{}",
                        self.get_deferred_program_name(),
                        dtype.short_name()
                    );

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
#include "cuda_fp16.h"

extern "C" __global__ void kernel(const size_t *info, const uint8_t *lhs, uint8_t *out) {{
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t byte_stride = info[2];
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {{
        auto x = *static_cast<{x_buf_ty} *>(lhs + i);
        {prog}

        TODO do sum

        *static_cast<{dst_ty} *>(out + i * byte_stride) = x;
    }}
}}
"#
                        );
                        let ptx = compile_ptx(kernel_src).unwrap();
                        cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                    }

                    let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                    let mut info = Vec::new();
                    info.push(y_numel);
                    let info = cuda.htod_copy(info)?;

                    unsafe {
                        fwd_fn.launch(
                            launch_cfg::<128>(y_numel as u32),
                            (&info, x_buf, &mut y_buf),
                        )
                    }?;

                    BytesPtr::Cuda(y_buf)
                }
                _ => unimplemented!(),
            };

            build_tensor(
                dtype,
                shape,
                strides,
                bytes,
                self.requires_grad() && crate::backward::is_recording(),
            )
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let y_grad = y_grad.broadcast_like(axis, &x_grad)?;
                x_grad.alloc()?.add_assign(y_grad)
            });
        }
        Ok(y)
    }
}
