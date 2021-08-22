//! Implementation of the MacKay-Neal CLI tool

use crate::cli::*;
use crate::mackay_neal::{Config, FillPolicy};
use std::error::Error;
use structopt::StructOpt;

/// MacKay-Neal CLI options
#[derive(Debug, StructOpt)]
#[structopt(about = "Generates LDPC codes using the MacKay-Neal algorithm")]
pub struct Opt {
    /// Number of rows
    num_rows: usize,
    /// Number of columns
    num_columns: usize,
    /// Maximum row weight
    wr: usize,
    /// Column weight
    wc: usize,
    /// Seed
    seed: u64,
    /// Columns to backtrack
    #[structopt(long, default_value = "0")]
    backtrack_cols: usize,
    /// Backtrack attemps
    #[structopt(long, default_value = "0")]
    backtrack_trials: usize,
    /// Minimum girth
    #[structopt(long)]
    min_girth: Option<usize>,
    /// Girth trials
    #[structopt(long, default_value = "0")]
    girth_trials: usize,
    /// Use uniform fill policy
    #[structopt(long)]
    uniform: bool,
    /// Maximum seed trials
    #[structopt(long, default_value = "1000")]
    seed_trials: u64,
    /// Maximum seed trials
    #[structopt(long)]
    search: bool,
}

impl Opt {
    fn config(&self) -> Config {
        Config {
            nrows: self.num_rows,
            ncols: self.num_columns,
            wr: self.wr,
            wc: self.wc,
            backtrack_cols: self.backtrack_cols,
            backtrack_trials: self.backtrack_trials,
            min_girth: self.min_girth,
            girth_trials: self.girth_trials,
            fill_policy: match self.uniform {
                true => FillPolicy::Uniform,
                false => FillPolicy::Random,
            },
        }
    }
}

impl Run for Opt {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let conf = self.config();
        let h = if self.search {
            let (seed, hh) = conf.search(self.seed, self.seed_trials);
            eprintln!("seed = {}", seed);
            hh
        } else {
            conf.run(self.seed)?
        };
        println!("{}", h.alist());
        Ok(())
    }
}
