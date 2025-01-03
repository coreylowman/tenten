use crate::{tensor::*, util::*};

impl Tensor {
    pub fn cos(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "cosf16",
                |a| f16::from_f32(a.as_f16().to_f32().cos()).into(),
                "hcos($x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "cosbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().cos()).into(),
                "hcos($x)",
            ),
            Dtype::Float32 => self.defer_op("cosf32", |a| a.as_f32().cos().into(), "cosf($x)"),
            Dtype::Float64 => self.defer_op("cosf64", |a| a.as_f64().cos().into(), "cos($x)"),
            _ => unimplemented!("Can't take cos of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.sin()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
