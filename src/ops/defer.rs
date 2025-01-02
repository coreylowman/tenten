use std::{cell::RefCell, rc::Rc};

use crate::tensor::*;

impl Tensor {
    #[inline(always)]
    pub fn defer_op(mut self, name: &str, cpu_op: fn(&Scalar) -> Scalar, cuda_op: &str) -> Self {
        self.deferred_ops.push(DeferredOp {
            name: name.into(),
            cpu_op: cpu_op.into(),
            cuda_op: cuda_op.into(),
        });
        if let Some(grad) = self.gradient.as_mut() {
            grad.bytes_ptr = Rc::new(RefCell::new(self.bytes_ptr.borrow().lazy()));
        }
        self
    }

    #[inline(always)]
    pub fn defer_op_with_args<Name: Into<String>, CudaOp: Into<String>>(
        mut self,
        name: Name,
        cpu_op: (fn(&Scalar, &[Scalar]) -> Scalar, Vec<Scalar>),
        cuda_op: CudaOp,
    ) -> Self {
        self.deferred_ops.push(DeferredOp {
            name: name.into(),
            cpu_op: cpu_op.into(),
            cuda_op: cuda_op.into(),
        });
        if let Some(grad) = self.gradient.as_mut() {
            grad.bytes_ptr = Rc::new(RefCell::new(self.bytes_ptr.borrow().lazy()));
        }
        self
    }

    pub fn get_deferred_program_name(&self) -> String {
        self.deferred_ops
            .iter()
            .map(|op| op.name.clone())
            .collect::<Vec<String>>()
            .join("-")
    }

    #[inline]
    pub fn deferred_ops_cpu_closure(&self) -> impl Fn(&Scalar) -> Scalar {
        let ops: Vec<CpuOpPtr> = self
            .deferred_ops
            .iter()
            .map(|op| op.cpu_op.clone())
            .collect();
        move |a| {
            let mut x = *a;
            for op in ops.iter() {
                x = match op {
                    CpuOpPtr::Simple(f) => f(&x),
                    CpuOpPtr::WithOptions(f, args) => f(&x, args),
                };
            }
            x
        }
    }
    pub fn deffered_ops_cuda_instructions(&self) -> String {
        let mut prog = String::new();
        for op in self.deferred_ops.iter() {
            if op.cuda_op.contains("=") {
                // for when the type changes
                prog += &std::format!("{};\n", op.cuda_op);
            } else {
                prog += &std::format!("x = {};\n", op.cuda_op);
            }
        }
        prog
    }
}
