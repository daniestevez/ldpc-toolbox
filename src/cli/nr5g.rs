//! 5G NR CLI subcommand.
//!
//! This subcommand can be used to generate the LDPC codes used in 5G NR. It
//! prints the alist of the parity matrix to `stdout`, and optionally computes
//! and prints the girth of the Tanner graph. Se [`crate::codes::nr5g`] for more
//! information about the 5G NR LPDC codes.
//!
//! # Examples
//! The base graph 1 with lifting size 32 can be generate with
//! ```shell
//! $ ldpc-toolbox 5g --base-graph 1 --lifting-size 32
//! ```
//! Its girth is computed with
//! ```shell
//! $ ldpc-toolbox 5g --base-graph 1 --lifting-size 32 --girth
//! ```

use crate::cli::*;
use crate::codes::nr5g::{BaseGraph, LiftingSize};
use clap::Parser;

/// 5G NR arguments.
#[derive(Debug, Parser)]
pub struct Args {
    /// Base graph
    #[arg(long)]
    base_graph: BaseGraph,
    /// Lifting size
    #[arg(long)]
    lifting_size: LiftingSize,
    /// Performs girth calculation
    #[arg(long)]
    girth: bool,
}

impl Run for Args {
    fn run(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let h = self.base_graph.h(self.lifting_size);
        if self.girth {
            if let Some(g) = h.girth() {
                println!("Code girth = {}", g);
            } else {
                println!("Code girth is infinite");
            }
        } else {
            println!("{}", h.alist());
        }
        Ok(())
    }
}
