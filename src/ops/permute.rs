use crate::tensor::*;

impl Tensor {
    pub fn permute(mut self, order: &[isize]) -> Result<Self, Error> {
        let num_dims = self.shape.len();

        assert_eq!(num_dims, order.len());
        let mut dup_found = false;
        for i in 0..num_dims {
            for j in i + 1..num_dims {
                if order[i] == order[j] {
                    dup_found = true;
                }
            }
        }
        assert!(
            !dup_found,
            "Must specify each dimension exactly once in permute command"
        );

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        for (i, new_dim) in order.iter().enumerate() {
            let new_dim = new_dim.rem_euclid(num_dims as isize) as usize;
            new_strides[i] = new_dim;
            new_shape[i] = self.shape[new_dim];
        }
        self.shape = new_shape;
        self.strides = new_strides;

        Ok(self)
    }
}
