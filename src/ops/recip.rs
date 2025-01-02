use crate::{tensor::*, util::*};

impl Tensor {
    pub fn recip(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "recipf16",
                |a| f16::from_f32(a.as_f16().to_f32().recip()).into(),
                "__half(1.0) / x",
            ),
            Dtype::BFloat16 => self.defer_op(
                "recipbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().recip()).into(),
                "__nv_bfloat16(1.0) / x",
            ),
            Dtype::Float32 => self.defer_op("recipf32", |a| a.as_f32().recip().into(), "1.0 / x"),
            Dtype::Float64 => self.defer_op("recipf64", |a| a.as_f64().recip().into(), "1.0 / x"),
            _ => unimplemented!("Can't take recip of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.recip()?.square()?.negate()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
