//! # LDPC toolbox
//!
//! `ldpc_toolbox` is a collection of Rust utilities to generate LDPC codes.
//! The goal is to eventually support several LDPC design algorithms from the
//! literature.

pub mod codes;
pub mod mackay_neal;
pub mod rand;
pub mod sparse;
