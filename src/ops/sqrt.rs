use crate::{tensor::*, util::*};

impl Tensor {
    pub fn sqrt(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "sqrtf16",
                |a| f16::from_f32(a.as_f16().to_f32().sqrt()).into(),
                "hsqrt(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "sqrtbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().sqrt()).into(),
                "hsqrt(x)",
            ),
            Dtype::Float32 => self.defer_op("sqrtf32", |a| a.as_f32().sqrt().into(), "sqrtf(x)"),
            Dtype::Float64 => self.defer_op("sqrtf64", |a| a.as_f64().sqrt().into(), "sqrt(x)"),
            _ => unimplemented!("Can't take sqrt of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.sqrt()?.mul_scalar(2.0f64)?.recip()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
