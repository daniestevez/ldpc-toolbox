//! # LDPC toolbox
//!
//! `ldpc_toolbox` is a collection of Rust utilities to generate LDPC codes.
//! The goal is to eventually support several LDPC design algorithms from the
//! literature.
//!
//! It can be used as a Rust library or as a CLI tool that allows access from
//! the command line to many of the algorithms implemented in `ldpc-toolbox`. See
//! [`cli`] for documentation about the usage of the CLI tool.

#![warn(missing_docs)]

mod c_api;
pub mod cli;
pub mod codes;
pub mod decoder;
pub mod encoder;
pub mod gf2;
pub mod mackay_neal;
pub mod peg;
pub mod rand;
pub mod simulation;
pub mod sparse;
pub mod systematic;

mod linalg;
mod util;
