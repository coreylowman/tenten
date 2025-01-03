use crate::{tensor::*, util::*};

impl Tensor {
    pub fn square(self) -> Result<Self, Error> {
        let dtype = self.dtype();
        let x = self.clone();
        let y = match dtype {
            Dtype::Float16 => self.defer_op(
                "squaref16",
                |a| f16::from_f32(a.as_f16().to_f32().powi(2)).into(),
                "$x*$x",
            ),
            Dtype::BFloat16 => self.defer_op(
                "squarebf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().powi(2)).into(),
                "$x*$x",
            ),
            Dtype::Float32 => self.defer_op("squaref32", |a| a.as_f32().powi(2).into(), "$x*$x"),
            Dtype::Float64 => self.defer_op("squaref64", |a| a.as_f64().powi(2).into(), "$x*$x"),
            Dtype::Int8 => self.defer_op("squarei8", |a| a.as_i8().pow(2).into(), "$x*$x"),
            Dtype::Int16 => self.defer_op("squarei16", |a| a.as_i16().pow(2).into(), "$x*$x"),
            Dtype::Int32 => self.defer_op("squarei32", |a| a.as_i32().pow(2).into(), "$x*$x"),
            Dtype::Int64 => self.defer_op("squarei64", |a| a.as_i64().pow(2).into(), "$x*$x"),
            Dtype::UInt8 => self.defer_op("squareu8", |a| a.as_u8().pow(2).into(), "$x*$x"),
            Dtype::UInt16 => self.defer_op("squareu16", |a| a.as_u16().pow(2).into(), "$x*$x"),
            Dtype::UInt32 => self.defer_op("squareu32", |a| a.as_u32().pow(2).into(), "$x*$x"),
            Dtype::UInt64 => self.defer_op("squareu64", |a| a.as_u64().pow(2).into(), "$x*$x"),
            _ => unimplemented!("Can't take square of dtype={dtype:?}"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.mul_scalar(2.0f64)?;
                x_grad.alloc()?.add_assign(dfdx.mul(y_grad)?)
            });
        }
        Ok(y)
    }
}
