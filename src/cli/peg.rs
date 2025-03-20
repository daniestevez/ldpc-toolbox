//! Progressive Edge Growth (PEG) CLI subcommand
//!
//! This uses the Progressive Edge Growth (PEG) pseudorandom construction to
//! build and LDPC parity check matrix. It runs the PEG algorithm and, if the
//! construction is successful, prints to `stout` the alist of the parity check
//! matrix. Optionally, it can also print to `stderr` the girth of the generated
//! code. For more details about this construction, see [`crate::peg`].
//!
//! # Examples
//! An r=1/2, n=16800 regular code with column weight 3 can be generated
//! with
//! ```shell
//! $ ldpc-toolbox peg 8100 16200 3 0
//! ```
//! To construct the code and only show the girth, we run
//! ```shell
//! $ ldpc-toolbox peg 8100 16200 3 0 --girth > /dev/null
//! ```

use crate::cli::*;
use crate::peg::Config;
use clap::Parser;
use std::error::Error;

/// PEG CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Generates LDPC codes using the Progressive Edge Growth algorithm")]
pub struct Args {
    /// Number of rows
    num_rows: usize,
    /// Number of columns
    num_columns: usize,
    /// Column weight
    wc: usize,
    /// Seed
    seed: u64,
    /// Performs girth calculation
    #[arg(long)]
    girth: bool,
}

impl Args {
    fn config(&self) -> Config {
        Config {
            nrows: self.num_rows,
            ncols: self.num_columns,
            wc: self.wc,
        }
    }
}

impl Run for Args {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let conf = self.config();
        let h = conf.run(self.seed)?;
        println!("{}", h.alist());
        if self.girth {
            match h.girth() {
                Some(g) => eprintln!("Code girth = {}", g),
                None => eprintln!("Code girth = infinity (there are no cycles)"),
            };
        }
        Ok(())
    }
}
