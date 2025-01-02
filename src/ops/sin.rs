use crate::{tensor::*, util::*};

impl Tensor {
    pub fn sin(self) -> Result<Self, Error> {
        let dtype = self.dtype();

        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "sinf16",
                |a| f16::from_f32(a.as_f16().to_f32().sin()).into(),
                "hsin(x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "sinbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().sin()).into(),
                "hsin(x)",
            ),
            Dtype::Float32 => self.defer_op("sinf32", |a| a.as_f32().sin().into(), "sinf(x)"),
            Dtype::Float64 => self.defer_op("sinf64", |a| a.as_f64().sin().into(), "sin(x)"),
            _ => unimplemented!("Can't take sin of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.cos()?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
