//! Design principles:
//! 1. Support sharding & DP/TP from ground up
//! 2. No multi thread support - assume multiple processes for everything.

mod backward;
mod cuda;
mod dtype;
mod init;
mod ops;
mod tensor;

pub use backward::*;
pub use cuda::*;
pub use dtype::*;
pub use init::*;
pub use ops::*;
pub use tensor::*;
