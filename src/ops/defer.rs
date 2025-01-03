use std::{cell::RefCell, ops::DerefMut, rc::Rc};

use cudarc::{driver::LaunchAsync, nvrtc::compile_ptx};

use crate::{tensor::*, util::launch_cfg};

impl Tensor {
    #[inline(always)]
    pub fn defer_op(mut self, name: &str, cpu_op: fn(&Scalar) -> Scalar, cuda_op: &str) -> Self {
        self.id = monotonically_increasing_id();
        self.deferred_ops.push(DeferredOp {
            name: name.into(),
            cpu_op: cpu_op.into(),
            cuda_op: cuda_op.into(),
        });
        self.set_new_grad();
        self
    }

    #[inline(always)]
    pub fn defer_op_with_args<Name: Into<String>, CudaOp: Into<String>>(
        mut self,
        name: Name,
        cpu_op: (fn(&Scalar, &[Scalar]) -> Scalar, Vec<Scalar>),
        cuda_op: CudaOp,
    ) -> Self {
        self.id = monotonically_increasing_id();
        self.deferred_ops.push(DeferredOp {
            name: name.into(),
            cpu_op: cpu_op.into(),
            cuda_op: cuda_op.into(),
        });
        self.set_new_grad();
        self
    }

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

    pub fn undeferred(mut self) -> Result<Self, Error> {
        if self.deferred_ops.is_empty() {
            return Ok(self);
        }

        assert!(self.deferred_dtype.num_bytes() <= self.byte_stride);

        let byte_stride = self.byte_stride;
        let stored_dtype = self.stored_dtype;
        let dtype = self.deferred_dtype;
        let numel = self.numel();

        let prog_name = self.get_deferred_program_name();
        let cpu_prog = self.deferred_ops_cpu_closure();
        let cuda_prog = self.deffered_ops_cuda_instructions();

        match Rc::make_mut(&mut self.bytes).borrow_mut().deref_mut() {
            BytesPtr::Cpu(buf) => {
                for i in (0..buf.len()).step_by(byte_stride) {
                    let value = stored_dtype.read(&buf[i..]);
                    cpu_prog(&value).store(&mut buf[i..]);
                }
            }
            BytesPtr::Cuda(buf) => {
                let cuda = buf.device();
                let src_ty = stored_dtype.cuda_type_name();
                let dst_ty = dtype.cuda_type_name();

                let module_name = std::format!("{}undefer{}", prog_name, dtype.short_name());

                if !cuda.has_func(&module_name, "kernel") {
                    let kernel_src = std::format!(
                        r#"
#include "cuda_fp16.h"
extern "C" __global__ void kernel(const size_t *info, uint8_t *buf) {{
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
                let info = cuda.htod_copy(vec![numel, byte_stride])?;
                unsafe { fwd_fn.launch(launch_cfg::<128>(numel as u32), (&info, buf)) }?;
            }
            _ => (),
        };

        self.deferred_ops.clear();
        self.stored_dtype = self.deferred_dtype;

        Ok(self)
    }
}
