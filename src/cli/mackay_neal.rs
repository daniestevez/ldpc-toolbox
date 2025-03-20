//! MacKay-Neal CLI subcommand
//!
//! This subcommand uses the MacKay-Neal pseudorandom construction to build
//! an LDPC parity check matrix. It runs the MacKay-Neal algorithm and,
//! if the construction is successful, prints to `stdout` the alist of the
//! parity check matrix. For more details about this construction, see
//! [`crate::mackay_neal`].
//!
//! # Examples
//! An r=1/2, n=16800 regular code with column weight 3 can be generated
//! with
//! ```shell
//! $ ldpc-toolbox mackay-neal 8100 16200 6 3 0 --uniform
//! ```
//! The `--uniform` parameter is useful when constructing regular codes
//! to prevent the construction from failing (see [`FillPolicy`]).
//!
//! A minimum graph girth can be enforced using the `--min_girth` and
//! `--girth_trials` parameters. For instance, a code of girth at least
//! 8 can be construced like so:
//! ```shell
//! $ ldpc-toolbox mackay-neal 8100 16200 6 3 0 --uniform \
//!       --min-girth 8 --girth-trials 1000
//! ```
//! This uses backtracking to try to find a construction that satisfies
//! the girth requirement.
//!
//! For high rate codes, the construction is less likely to suceed even if
//! backtracking is used. The `--search` parameter is useful to try several
//! seeds in parallel. It will print to `stderr` the seed that gave a successful
//! construction. For instance, an r=8/9 code with no 4-cycles can be
//! constructed with
//! ```shell
//! $ ldpc-toolbox mackay-neal 1800 16200 27 3 0 --uniform \
//!       --min-girth 6 --girth-trials 1000 --search
//! ```

use crate::cli::*;
use crate::mackay_neal::{Config, FillPolicy};
use clap::Parser;
use std::error::Error;

/// MacKay-Neal CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Generates LDPC codes using the MacKay-Neal algorithm")]
pub struct Args {
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
    #[arg(long, default_value_t = 0)]
    backtrack_cols: usize,
    /// Backtrack attemps
    #[arg(long, default_value_t = 0)]
    backtrack_trials: usize,
    /// Minimum girth
    #[arg(long)]
    min_girth: Option<usize>,
    /// Girth trials
    #[arg(long, default_value_t = 0)]
    girth_trials: usize,
    /// Use uniform fill policy
    #[arg(long)]
    uniform: bool,
    /// Maximum seed trials
    #[arg(long, default_value_t = 1000)]
    seed_trials: u64,
    /// Try several seeds in parallel
    #[arg(long)]
    search: bool,
}

impl Args {
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

impl Run for Args {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let conf = self.config();
        let h = if self.search {
            let (seed, hh) = conf
                .search(self.seed, self.seed_trials)
                .ok_or("no solution found")?;
            eprintln!("seed = {}", seed);
            hh
        } else {
            conf.run(self.seed)?
        };
        println!("{}", h.alist());
        Ok(())
    }
}
