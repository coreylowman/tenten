use crate::tensor::*;

impl Tensor {
    pub fn all_close<S: Into<Scalar>>(
        self,
        other: Tensor,
        rtol: S,
        atol: S,
        nans_equal: bool,
    ) -> Result<bool, Error> {
        let no_grad = crate::backward::no_grad();
        let cond = self
            .sub(other.clone())?
            .abs()?
            .le(other.abs()?.mul_scalar(rtol)?.add_scalar(atol)?)?;
        drop(no_grad);
        todo!("reduce the condition tensor?")
    }
}
