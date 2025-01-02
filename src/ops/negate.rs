use std::ops::Neg;

use crate::{tensor::*, util::*};

impl Tensor {
    pub fn negate(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "negatef16",
                |a| f16::from_f32(a.as_f16().to_f32().neg()).into(),
                "-x",
            ),
            Dtype::BFloat16 => self.defer_op(
                "negatebf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().neg()).into(),
                "-x",
            ),
            Dtype::Float32 => self.defer_op("negatef32", |a| a.as_f32().neg().into(), "-x"),
            Dtype::Float64 => self.defer_op("negatef64", |a| a.as_f64().neg().into(), "-x"),
            Dtype::Int8 => self.defer_op("negatei8", |a| a.as_i8().neg().into(), "-x"),
            Dtype::Int16 => self.defer_op("negatei16", |a| a.as_i16().neg().into(), "-x"),
            Dtype::Int32 => self.defer_op("negatei32", |a| a.as_i32().neg().into(), "-x"),
            Dtype::Int64 => self.defer_op("negatei64", |a| a.as_i64().neg().into(), "-x"),
            _ => unimplemented!("Can't take negate of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || x_grad.alloc()?.add_assign(y_grad.negate()?));
        }
        Ok(y)
    }
}
