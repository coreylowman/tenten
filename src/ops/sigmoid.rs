use std::ops::Neg;

use crate::{tensor::*, util::*};

fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + x.neg().exp())
}

impl Tensor {
    pub fn sigmoid(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "sigmoidf16",
                |a| f16::from_f32(sigmoidf(a.as_f16().to_f32())).into(),
                "__half(1.0) / (__half(1.0) + hexp(-$x))",
            ),
            Dtype::BFloat16 => self.defer_op(
                "sigmoidbf16",
                |a| bf16::from_f32(sigmoidf(a.as_bf16().to_f32())).into(),
                "__nv_bfloat16(1.0) / (__nv_bfloat16(1.0) + hexp(-$x))",
            ),
            Dtype::Float32 => self.defer_op(
                "sigmoidf32",
                |a| sigmoidf(a.as_f32()).into(),
                "1.0 / (1.0 + expf(-$x))",
            ),
            Dtype::Float64 => self.defer_op(
                "sigmoidf64",
                |a| (1.0 / (1.0 + a.as_f64().neg().exp())).into(),
                "1.0 / (1.0 + exp(-$x))",
            ),
            _ => unimplemented!("Can't take sigmoid of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let y = x.sigmoid()?;
                let dfdx = y.clone().mul(y.negate()?.add_scalar(dtype.one())?)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
