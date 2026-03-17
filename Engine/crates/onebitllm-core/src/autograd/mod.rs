pub mod tape;
pub mod variable;
pub mod ops;
pub mod backward;

pub use variable::{Variable, VarId};
pub use tape::Tape;
