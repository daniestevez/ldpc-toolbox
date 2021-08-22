//! Implementation of the Progressive Edge Growth (PEG) CLI tool

use crate::cli::*;
use crate::peg::Config;
use std::error::Error;
use structopt::StructOpt;

/// PEG CLI options
#[derive(Debug, StructOpt)]
#[structopt(about = "Generates LDPC codes using the Progressive Edge Growth algorithm")]
pub struct Opt {
    /// Number of rows
    num_rows: usize,
    /// Number of columns
    num_columns: usize,
    /// Column weight
    wc: usize,
    /// Seed
    seed: u64,
}

impl Opt {
    fn config(&self) -> Config {
        Config {
            nrows: self.num_rows,
            ncols: self.num_columns,
            wc: self.wc,
        }
    }
}

impl Run for Opt {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let conf = self.config();
        let h = conf.run(self.seed)?;
        println!("{}", h.alist());
        println!("Girth = {:?}", h.girth());
        Ok(())
    }
}
