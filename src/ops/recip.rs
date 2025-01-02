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

#[cfg(test)]
mod tests {
    use crate::{init::*, tensor::*, tests::*};

    #[test]
    fn test_recip() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);
        let x = Tensor::from([1.0, 0.5, 2.0]).to_dtype(TEST_DTYPE)?;
        assert_all_close(
            &x.recip()?.to_dtype(Dtype::Float32)?.into_vec()?,
            &[1.0f32, 2.0, 0.5],
        )
    }
}
