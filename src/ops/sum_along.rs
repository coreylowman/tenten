use std::ops::Deref;

use cudarc::driver::LaunchAsync;

use crate::{init::build_tensor, tensor::*, util::*};

impl Tensor {
    pub fn sum_along<A: Into<Axis>>(mut self, axis: A) -> Result<Self, Error> {
        let axis = Into::<Axis>::into(axis);
        let ax = axis.to_usize(self.shape.len());
        let x = self.clone();
        let y = if self.shape[ax] == 1 {
            self.id = monotonically_increasing_id();
            self.shape.remove(ax);
            self.strides.remove(ax);
            self.set_new_grad();
            self
        } else if self.strides[ax] == 0 {
            self.id = monotonically_increasing_id();
            let scale = self.shape.remove(ax);
            self.strides.remove(ax);
            self.set_new_grad();
            self.mul_scalar(scale)?
        } else {
            let dtype = self.deferred_dtype;
            let mut shape = self.shape.clone();
            shape.remove(ax);
            let strides = crate::init::nd_bytes_strides(&shape, dtype.num_bytes());
            let x_numel = self.numel();
            let y_numel: usize = shape.iter().product();
            let y_byte_stride = dtype.num_bytes();
            let y_num_bytes = y_numel * y_byte_stride;
            let num_to_sum = self.shape[ax];

            let bytes = match self.bytes.borrow().deref() {
                BytesPtr::Cpu(x_buf) => {
                    let prog = self.build_cpu_op();

                    let mut y_buf = vec![0; y_num_bytes];

                    let mut idx = CpuIndex::new(&self.shape, &self.strides, self.byte_stride);

                    for i_y in (0..y_num_bytes).step_by(dtype.num_bytes()) {
                        // TODO do accumulation in f32 for f16/bf16
                        let mut y = dtype.zero();
                        for _ in 0..num_to_sum {
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

                    let prog = self.build_cuda_op("x", "xf");
                    let src_ty = self.stored_dtype.cuda_type_name();
                    let dst_ty = dtype.cuda_type_name();

                    let mut y_buf = cuda.alloc_zeros(y_num_bytes)?;

                    let module_name =
                        std::format!("{}sum{ax:?}{}", self.build_op_name(), dtype.short_name());

                    const BLOCK_SIZE: u32 = 1024;

                    if !cuda.has_func(&module_name, "kernel") {
                        let kernel_src = std::format!(
                            r#"
typedef unsigned char uint8_t;
#include "cuda_fp16.h"

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
    const size_t num_to_sum = info[2];
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

    shr[tid] = 0;

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
        shr[tid] += xf;

        idx += blockDim.x;
    }}

    // Parallel reduction, see the slides:
    // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
    // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        __syncthreads();
        if (tid < s) {{
            shr[tid] += shr[tid + s];
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
                    info.push(num_to_sum);
                    info.push(y_byte_stride);
                    info.push(self.shape.len());
                    info.extend(&self.shape);
                    info.extend(&self.strides);
                    let info = cuda.htod_copy(info)?;

                    let block_dim = usize::min(1024, num_to_sum).next_power_of_two();
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
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let y_grad = y_grad.broadcast_like(axis, &x_grad)?;
                x_grad.alloc()?.add_assign(y_grad)
            });
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use crate::{init::*, tensor::*, tests::*};

    #[test]
    fn test_sum_along_contiguous() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);
        let x = Tensor::from([
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [-1.0, 2.0, -3.0, 4.0, -5.0],
            [1.0, -2.0, 3.0, -4.0, 5.0],
        ]);

        let y = x
            .to_dtype(TEST_DTYPE)?
            .sum_along(-1)?
            .to_dtype(Dtype::Float32)?;

        assert_eq!(y.shape, [3]);

        assert_all_close(&y.into_vec()?, &[15.0, -3.0, 3.0])
    }

    #[test]
    fn test_sum_along_broadcasted() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);

        let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0, 5.0]);

        let y = x
            .to_dtype(TEST_DTYPE)?
            .broadcast_along(-1, 3)?
            .sum_along(-1)?
            .to_dtype(Dtype::Float32)?;

        assert_eq!(y.shape, [5]);

        assert_all_close(&y.into_vec()?, &[3.0, 6.0, 9.0, 12.0, 15.0])
    }

    #[test]
    fn test_sum_along_permuted() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);

        let x = Tensor::from([
            [1.0f32, 2.0, 3.0, 4.0, 5.0],
            [-1.0, 2.0, -3.0, 4.0, -5.0],
            [1.0, -2.0, 3.0, -4.0, 5.0],
        ]);

        let y = x
            .to_dtype(TEST_DTYPE)?
            .permute([1, 0])?
            .sum_along(-1)?
            .to_dtype(Dtype::Float32)?;

        assert_eq!(y.shape, [5]);

        assert_all_close(&y.into_vec()?, &[1.0, 2.0, 3.0, 4.0, 5.0])
    }

    #[test]
    fn test_sum_along_smaller_than_block_size() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);

        let x = full([256, 512], TEST_DTYPE.one())?;

        assert_all_close(
            &x.clone()
                .sum_along(0)?
                .to_dtype(Dtype::Float32)?
                .into_vec()?,
            &[256.0; 512],
        )?;

        assert_all_close(
            &x.sum_along(1)?.to_dtype(Dtype::Float32)?.into_vec()?,
            &[512.0; 256],
        )
    }

    #[test]
    fn test_sum_along_larger_than_block_size() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);

        let x = full([1024, 2048], TEST_DTYPE.one())?;

        assert_all_close(
            &x.clone()
                .sum_along(0)?
                .to_dtype(Dtype::Float32)?
                .into_vec()?,
            &[1024.0; 2048],
        )?;

        assert_all_close(
            &x.sum_along(1)?.to_dtype(Dtype::Float32)?.into_vec()?,
            &[2048.0; 1024],
        )
    }
}
