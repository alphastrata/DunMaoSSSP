#![allow(deprecated)]
pub mod graph;
#[deprecated(
    note = "Use sequential_v2, this only remains as a reference of my implementation crimes"
)]
pub mod sequential;
pub mod sequential_v2;
pub mod utils;

#[deprecated(note = "use parallel_v2!")]
#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "parallel")]
pub mod parallel_v2;

#[cfg(feature = "petgraph")]
pub mod petgraph_utils;

pub use graph::{Edge, Graph};
#[deprecated(
    note = "Use sequential_v2, this only remains as a reference of my implementation crimes"
)]
pub use sequential::SSSpSolver;

pub use sequential_v2::DuanMaoSolverV2;
