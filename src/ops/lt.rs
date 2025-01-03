use std::ops::Deref;

use cudarc::driver::LaunchAsync;

use crate::{init::build_tensor, tensor::*, util::*};

impl Tensor {
    pub fn lt(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            return crate::init::full(self.shape(), true)?.to_device(self.device());
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.shape(), other.shape());
        assert_eq!(self.device(), other.device());

        let dtype = Dtype::Boolean;
        let numel = self.numel();
        let tensor_num_bytes = numel * dtype.num_bytes();

        let shape = &self.shape;

        let byte_stride = dtype.num_bytes();
        let z_strides = crate::init::nd_bytes_strides(shape, byte_stride);

        let bytes = match (self.bytes.borrow().deref(), other.bytes.borrow().deref()) {
            (&BytesPtr::Ghost(d, l), _) => BytesPtr::Ghost(d, l),
            (BytesPtr::Cpu(x_buf), BytesPtr::Cpu(y_buf)) => {
                let x_prog = self.build_cpu_op();
                let y_prog = other.build_cpu_op();

                let mut z_buf = vec![0; tensor_num_bytes];

                let mut x_idx = CpuIndex::new(shape, &self.strides, self.byte_stride);
                let mut y_idx = CpuIndex::new(shape, &other.strides, other.byte_stride);
                for i_out in 0..numel {
                    let i_lhs = x_idx.next().unwrap();
                    let i_rhs = y_idx.next().unwrap();

                    let x = self.stored_dtype.read(&x_buf[i_lhs..]);
                    let y = other.stored_dtype.read(&y_buf[i_rhs..]);
                    let z = Scalar::from(x_prog(&x) < y_prog(&y));

                    z.store(&mut z_buf[i_out * byte_stride..]);
                }
                BytesPtr::Cpu(z_buf)
            }
            (BytesPtr::Cuda(x_buf), BytesPtr::Cuda(y_buf)) => {
                let x_buf_ty = self.stored_dtype.cuda_type_name();
                let y_buf_ty = other.stored_dtype.cuda_type_name();
                let dst_ty = dtype.cuda_type_name();
                let x_prog = self.build_cuda_op("x", "xf");
                let y_prog = other.build_cuda_op("y", "yf");

                let cuda = x_buf.device();

                let mut z_buf = cuda.alloc_zeros::<u8>(tensor_num_bytes)?;

                let module_name = std::format!(
                    "{}{}lt{}",
                    self.build_op_name(),
                    other.build_op_name(),
                    dtype.short_name()
                );

                if !cuda.has_func(&module_name, "kernel") {
                    let kernel_src = std::format!(
                        r#"
extern "C" __global__ void kernel(const size_t *info, const uint8_t *lhs, const uint8_t *rhs, uint8_t *out) {{
    const size_t numel = info[0];
    const size_t num_dims = info[1];
    const size_t byte_stride = info[2];
    const size_t *dims = info + 3;
    const size_t *lhs_strides = info + 3 + num_dims;
    const size_t *rhs_strides = info + 3 + 2 * num_dims;
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

        auto x = *reinterpret_cast<const {x_buf_ty} *>(lhs + lhs_i);
        {x_prog}
        auto y = *reinterpret_cast<const {y_buf_ty} *>(rhs + rhs_i);
        {y_prog}
        *reinterpret_cast<{dst_ty} *>(out + i * byte_stride) = xf < yf;
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
                info.push(dtype.num_bytes());
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

        Ok(build_tensor(dtype, shape.clone(), z_strides, bytes, false))
    }
}
