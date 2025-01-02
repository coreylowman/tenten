use crate::tensor::*;

impl Tensor {
    pub fn dot(self, other: Self) -> Result<Self, Error> {
        if self.is_same_as(&other) {
            todo!()
        }

        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.device(), other.device());

        let lhs_shape = self.shape();
        let rhs_shape = other.shape();

        // dim checks
        match (&lhs_shape[..], &rhs_shape[..]) {
            ([m], [n]) => (),
            ([k1], [k2, n]) => assert_eq!(k1, k2),
            ([m, k1], [k2]) => assert_eq!(k1, k2),
            ([m, k1], [k2, n]) => assert_eq!(k1, k2),
            ([b, m, k1], [k2, n]) => assert_eq!(k1, k2),
            ([b1, m, k1], [b2, k2, n]) => {
                assert_eq!(b1, b2);
                assert_eq!(k1, k2);
            }
            ([b1, s1, m, k1], [b2, s2, k2, n]) => {
                assert_eq!(b1, b2);
                assert_eq!(s1, s2);
                assert_eq!(k1, k2);
            }
            (a, b) => unimplemented!("Unable to run dot product on shapes {a:?}x{b:?}"),
        };
        todo!()
    }
}
