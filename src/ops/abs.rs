use crate::{tensor::*, util::*};

impl Tensor {
    pub fn abs(self) -> Result<Self, Error> {
        let dtype = self.dtype();

        let x = self.clone();
        let y = match dtype {
            Dtype::Boolean => unimplemented!("Can't take abs of boolean tensor"),
            Dtype::UInt8 | Dtype::UInt16 | Dtype::UInt32 | Dtype::UInt64 => return Ok(self),
            Dtype::Float16 => self.defer_op(
                "absf16",
                |a| f16::from_f32(a.as_f16().to_f32().abs()).into(),
                "__habs($x)",
            ),
            Dtype::BFloat16 => self.defer_op(
                "absbf16",
                |a| bf16::from_f32(a.as_bf16().to_f32().abs()).into(),
                "__habs($x)",
            ),
            Dtype::Float32 => self.defer_op("absf32", |a| a.as_f32().abs().into(), "fabsf($x)"),
            Dtype::Float64 => self.defer_op("absf64", |a| a.as_f64().abs().into(), "fabs($x)"),
            Dtype::Int8 => self.defer_op("absi8", |a| a.as_i8().abs().into(), "abs($x)"),
            Dtype::Int16 => self.defer_op("absi16", |a| a.as_i16().abs().into(), "abs($x)"),
            Dtype::Int32 => self.defer_op("absi32", |a| a.as_i32().abs().into(), "abs($x)"),
            Dtype::Int64 => self.defer_op("absi64", |a| a.as_i64().abs().into(), "abs($x)"),
        };
        if let Some([x_grad, y_grad]) = all_some([x.grad(), y.grad()]) {
            crate::backward::record_op(move || {
                let dfdx = x.sign()?;
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
    fn test_abs() -> Result<(), Error> {
        set_default_device(TEST_DEVICE);

        let x = Tensor::from([
            -1.0,
            0.0,
            -0.0,
            1.0,
            TestDtype::INFINITY,
            TestDtype::NEG_INFINITY,
            TestDtype::NAN,
        ]);
        assert_all_close(
            &x.abs()?.into_vec()?,
            &[
                1.0,
                0.0,
                0.0,
                1.0,
                TestDtype::INFINITY,
                TestDtype::INFINITY,
                TestDtype::NAN,
            ],
        )
    }
}
