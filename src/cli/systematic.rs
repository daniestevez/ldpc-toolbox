//! Systematic CLI subcommand.
//!
//! This command can be used to convert an n x m parity check matrix into one
//! that supports systematic encoding using the first m - n columns by permuting
//! columns in such a way that the n x n submatrix formed by the last n columns
//! is invertible.

use crate::{cli::Run, sparse::SparseMatrix, systematic::parity_to_systematic};
use clap::Parser;
use std::error::Error;

/// Systematic CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Converts a parity check matrix into systematic form")]
pub struct Args {
    /// alist file for the code
    pub alist: String,
}

impl Run for Args {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let h = SparseMatrix::from_alist(&std::fs::read_to_string(&self.alist)?)?;
        let h_sys = parity_to_systematic(&h)?;
        println!("{}", h_sys.alist());
        Ok(())
    }
}
