//! # Reproducible random functions
//!
//! This module uses the [`ChaCha8Rng`] RNG from the [rand_chacha] crate
//! to achieve reproducible random number generation.
//!
//! # Examples
//! ```
//! # use ldpc_toolbox::rand::Rng;
//! # use ldpc_toolbox::rand::*;
//! let seed = 42;
//! let mut rng = Rng::seed_from_u64(seed);
//! assert_eq!(rng.next_u64(), 12578764544318200737);
//! ```
use rand_chacha::ChaCha8Rng;
pub use rand_chacha::rand_core::SeedableRng;
pub use rand_core::RngCore;

/// The RNG used in throughout this crate for algorithms using pseudorandom
/// generation.
pub type Rng = ChaCha8Rng;
