use crate::tensor::*;

impl Tensor {
    pub fn log_softmax_along<A: Into<Axis>>(mut self, axis: A) -> Result<Self, Error> {
        let axis = axis.into();

        let no_grad = crate::backward::no_grad();
        let max = self.clone().max_along(axis)?.broadcast_like(axis, &self)?;
        self.sub_assign(max)?;
        drop(no_grad);

        let logsumexp = self
            .clone()
            .exp()?
            .sum_along(axis)?
            .ln()?
            .broadcast_like(axis, &self)?;
        self.sub(logsumexp)
    }
}
