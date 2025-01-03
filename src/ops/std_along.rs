use crate::tensor::*;

impl Tensor {
    pub fn std_along<A: Into<Axis>, S: Into<Scalar>>(self, axis: A, eps: S) -> Result<Self, Error> {
        self.var_along(axis)?.add_scalar(eps)?.sqrt()
    }
}
