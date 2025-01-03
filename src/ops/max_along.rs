use std::ops::Deref;

use cudarc::driver::LaunchAsync;

use crate::{init::build_tensor, tensor::*, util::*};

impl Tensor {
    pub fn max_along<A: Into<Axis>>(mut self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let ax = axis.to_usize(self.shape.len());
        let x = self.clone();
        let y = if self.shape[ax] == 1 || self.strides[ax] == 0 {
            self.id = monotonically_increasing_id();
            self.shape.remove(ax);
            self.strides.remove(ax);
            self.set_new_grad();
            self
        } else {
            let dtype = self.deferred_dtype;
            let mut shape = self.shape.clone();
            shape.remove(ax);
            let strides = crate::init::nd_bytes_strides(&shape, dtype.num_bytes());
            let y_numel: usize = shape.iter().product();
            let y_byte_stride = dtype.num_bytes();
            let y_num_bytes = y_numel * y_byte_stride;
            let num_to_max = self.shape[ax];

            let bytes = match self.bytes.borrow().deref() {
                BytesPtr::Cpu(x_buf) => {
                    let prog = self.build_cpu_op();

                    let mut y_buf = vec![0; y_num_bytes];

                    let mut idx = CpuIndex::new(&self.shape, &self.strides, self.byte_stride);

                    for i_y in (0..y_num_bytes).step_by(dtype.num_bytes()) {
                        let mut y = dtype.min_value();
                        for _ in 0..num_to_max {
                            let i_x = idx.next().unwrap();
                            let x = self.stored_dtype.read(&x_buf[i_x..]);
                            let xf = prog(&x);
                            if xf > y {
                                y = xf;
                            }
                        }
                        y.store(&mut y_buf[i_y..]);
                    }

                    assert!(idx.next().is_none());

                    BytesPtr::Cpu(y_buf)
                }
                BytesPtr::Cuda(x_buf) => {
                    let cuda = x_buf.device();

                    let prog = self.build_cuda_op("x", "xf");
                    let src_ty = self.stored_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();
                    let min_value = dtype.min_value().to_string();
                    let max_fn = match dtype {
                        Dtype::Float16 => "__hmax_nan",
                        Dtype::BFloat16 => "__hmax_nan",
                        Dtype::Float32 => "fmaxf",
                        Dtype::Float64 => "fmax",
                        Dtype::Int8 => "max",
                        Dtype::Int16 => "max",
                        Dtype::Int32 => "max",
                        Dtype::Int64 => "max",
                        Dtype::UInt8 => "max",
                        Dtype::UInt16 => "max",
                        Dtype::UInt32 => "max",
                        Dtype::UInt64 => "max",
                        _ => unimplemented!(),
                    };

                    let mut y_buf = cuda.alloc_zeros(y_num_bytes)?;

                    let module_name =
                        std::format!("{}max{ax:?}{}", self.build_op_name(), dtype.short_name());

                    const BLOCK_SIZE: usize = 1024;

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
__device__ unsigned int get_strided_index(
    size_t idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {{
    size_t strided_i = 0;
    for (int d = num_dims - 1; d >= 0; d--) {{
        strided_i += (idx % dims[d]) * strides[d];
        idx /= dims[d];
    }}
    return strided_i;
}}

extern "C" __global__ void kernel(const size_t *info, const uint8_t *src, uint8_t *dst) {{
    const size_t src_numel = info[0];
    const size_t dst_numel = info[1];
    const size_t num_to_max = info[2];
    const size_t dst_byte_stride = info[3];
    const size_t src_num_dims = info[4];
    const size_t *src_dims = info + 5;
    const size_t *src_strides = info + 5 + src_num_dims;

    __shared__ {dst_ty} shr[{BLOCK_SIZE}];
    size_t tid = threadIdx.x;
    size_t dst_id = blockIdx.x;

    if (dst_id >= dst_numel) {{
        return;
    }}

    shr[tid] = {min_value};

    // Elements summed in this block range from dst_id * num_to_sum
    // to (dst_id + 1) * num_to_sum.
    size_t start_idx = dst_id * num_to_sum;
    size_t stop_idx = min((dst_id + 1) * num_to_sum, src_numel);
    size_t idx = start_idx + tid;

    while (idx < stop_idx) {{
        // TODO: Fast version for the contiguous case.
        size_t i = get_strided_index(idx, src_num_dims, src_dims, src_strides);
        auto x = *reinterpret_cast<const {src_ty} *>(src + i);
        {prog}
        shr[tid] = {max_fn}(shr[tid], xf);

        idx += blockDim.x;
    }}

    // Parallel reduction, see the slides:
    // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
    // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        __syncthreads();
        if (tid < s) {{
            shr[tid] = {max_fn}(shr[tid], shr[tid + s]);
        }}
    }}

    if (tid == 0) {{
        *reinterpret_cast<{dst_ty} *>(dst + dst_id * dst_byte_stride) = shr[0];
    }}
}}
"#
                        );
                        let ptx = jit_compile(kernel_src)?;
                        cuda.load_ptx(ptx, &module_name, &["kernel"])?;
                    }

                    let fwd_fn = cuda.get_func(&module_name, "kernel").unwrap();

                    let mut info = Vec::new();
                    info.push(self.numel());
                    info.push(y_numel);
                    info.push(num_to_max);
                    info.push(y_byte_stride);
                    info.push(self.shape.len());
                    info.extend(&self.shape);
                    info.extend(&self.strides);
                    let info = cuda.htod_copy(info)?;

                    let block_dim = usize::min(BLOCK_SIZE, num_to_max).next_power_of_two();
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: (y_numel as u32, 1, 1),
                        block_dim: (block_dim as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };

                    unsafe { fwd_fn.launch(cfg, (&info, x_buf, &mut y_buf)) }?;

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
        let yc = y.clone();
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let y_grad = y_grad.broadcast_like(axis, &x_grad)?;
                // TODO optimize these kernels
                let mask = yc
                    .broadcast_like(axis, &x)?
                    .eq(x)?
                    .to_dtype(x_grad.dtype())?;
                x_grad.alloc()?.add_assign(y_grad.mul(mask)?)
            });
        }
        Ok(y)
    }
}
