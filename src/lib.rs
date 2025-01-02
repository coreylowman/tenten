pub mod backward;
pub mod init;
pub mod losses;
pub mod ops;
pub mod tensor;
pub mod util;

#[cfg(test)]
pub(crate) mod tests {
    #[allow(unused)]
    use crate::tensor::{bf16, f16, Device, Dtype};

    #[cfg(not(any(feature = "test-f16", feature = "test-bf16")))]
    pub const TEST_DTYPE: Dtype = Dtype::Float32;
    #[cfg(feature = "test-f16")]
    pub const TEST_DTYPE: Dtype = Dtype::Float16;
    #[cfg(feature = "test-bf16")]
    pub const TEST_DTYPE: Dtype = Dtype::BFloat16;

    #[cfg(not(any(feature = "test-f16", feature = "test-bf16")))]
    pub type TestDtype = f32;
    #[cfg(feature = "test-f16")]
    pub type TestDtype = f16;
    #[cfg(feature = "test-bf16")]
    pub type TestDtype = bf16;

    #[cfg(not(any(feature = "test-cuda")))]
    pub const TEST_DEVICE: Device = Device::Cpu;
    #[cfg(feature = "test-cuda")]
    pub const TEST_DEVICE: Device = Device::Cuda(0);

    pub const RTOL: TestDtype = 1e-5;
    pub const ATOL: TestDtype = 1e-8;

    pub fn assert_all_close(a: &[TestDtype], b: &[TestDtype]) -> Result<(), crate::tensor::Error> {
        for (x, y) in a.iter().zip(b.iter()) {
            let close = if x.is_nan() {
                y.is_nan()
            } else if x.is_infinite() {
                y.is_infinite() && x.signum() == y.signum()
            } else {
                y.is_finite() && (x - y).abs() <= ATOL + RTOL * y.abs()
            };
            if !close {
                panic!("diff between {a:?} {b:?}");
            }
        }
        Ok(())
    }
}
