use crate::tensor::*;

impl Tensor {
    pub fn logsumexp_along<A: Into<Axis>>(mut self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();

        let no_grad = crate::backward::no_grad();
        let max = self.clone().max_along(axis)?;
        self.sub_assign(max.clone().broadcast_like(axis, &self)?)?;
        drop(no_grad);

        let mut x = self.exp()?.sum_along(axis)?.ln()?;

        let no_grad = crate::backward::no_grad();
        x.add_assign(max)?;
        drop(no_grad);

        Ok(x)
    }
}
