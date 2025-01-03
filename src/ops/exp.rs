use crate::{tensor::*, util::*};

impl Tensor {
    pub fn exp(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "expf16",
                |a| f16::from_f32(a.as_f16().to_f32().exp()).into(),
                "hexp($x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "expbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().exp()).into(),
                "hexp($x)",
            ),
            Dtype::Float32 => self.defer_op("expf32", |a| a.as_f32().exp().into(), "expf($x)"),
            Dtype::Float64 => self.defer_op("expf64", |a| a.as_f64().exp().into(), "exp($x)"),
            _ => unimplemented!("Can't take exp of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.exp()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
