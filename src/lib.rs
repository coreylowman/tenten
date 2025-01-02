//! Design principles:
//! 1. Support sharding & DP/TP from ground up
//! 2. No multi thread support - assume multiple processes for everything.

pub mod backward;
pub mod init;
pub mod losses;
pub mod ops;
pub mod tensor;
pub mod util;
