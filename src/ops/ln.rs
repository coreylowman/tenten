use crate::{tensor::*, util::*};

impl Tensor {
    pub fn ln(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "lnf16",
                |a| f16::from_f32(a.as_f16().to_f32().ln()).into(),
                "hlog($x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "lnbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().ln()).into(),
                "hlog($x)",
            ),
            Dtype::Float32 => self.defer_op("lnf32", |a| a.as_f32().ln().into(), "logf($x)"),
            Dtype::Float64 => self.defer_op("lnf64", |a| a.as_f64().ln().into(), "log($x)"),
            _ => unimplemented!("Can't take ln of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.recip()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
