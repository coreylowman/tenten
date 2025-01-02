use crate::{tensor::*, util::*};

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
