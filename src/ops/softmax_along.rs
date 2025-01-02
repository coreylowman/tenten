use crate::tensor::*;

impl Tensor {
    pub fn softmax_along<A: Into<Axis>>(mut self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();

        let no_grad = crate::backward::no_grad();
        let max = self.clone().max_along(axis)?.broadcast_like(axis, &self)?;
        self.sub_assign(max)?;
        drop(no_grad);

        let x_exp = self.clone().exp()?;
        let x_expsum = x_exp
            .clone()
            .sum_along(axis)?
            .broadcast_like(axis, &x_exp)?;
        x_exp.div(x_expsum)
    }
}
