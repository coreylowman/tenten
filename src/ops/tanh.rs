use crate::{tensor::*, util::*};

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
