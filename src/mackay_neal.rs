//! # MacKay-Neal pseudorandom LDPC construction
//!
//! This implements the algorithms from *MacKay, D.J. and Neal, R.M., 1996.
//! Near Shannon limit performance of low density parity check codes.
//! Electronics letters, 32(18), p.1645.* and variations on this idea.
//!
//! The algorithm works by adding column by column to the parity check
//! matrix. At each step, `wc` rows from the subset of rows that have not yet
//! achieved the total row weight `wr` are random chosen, and ones are inserted
//! in those positions.
//!
//! Optionally, to enforce a minimum girth, at each step the candidate
//! column is checked to see if it maintains the girth of the graph at or above
//! the minimum. If not, another random candidate column is
//! chosen according to the available rows. The algorithm aborts if after
//! a fixed number of trials it is unable to yield a new column satisfying
//! the required properties.
//!
//! # Examples
//! To run a MacKay-Neal LDPC generation algorithm, it is necessary
//! to create a [`Config`] and then use the `run()` method.
//! ```
//! # use ldpc_toolbox::mackay_neal::{Config, FillPolicy};
//! let conf = Config {
//!     nrows: 4,
//!     ncols: 8,
//!     wr: 4,
//!     wc: 2,
//!     backtrack_cols: 0,
//!     backtrack_trials: 0,
//!     min_girth: None,
//!     girth_trials: 0,
//!     fill_policy: FillPolicy::Uniform,
//!  };
//!  let seed = 42;
//!  let h = conf.run(seed).unwrap();
//!  print!("{}", h.alist());
//!  ```

use crate::rand::{Rng, *};
use crate::sparse::{Node, SparseMatrix};
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use std::fmt;
use std::fmt::{Display, Formatter};

/// Runtime errors of the MacKay-Neal construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// No rows available.
    NoAvailRows,
    /// Girth is too small (should not be returned to the user).
    GirthTooSmall,
    /// Exceeded backtrack trials.
    NoMoreBacktrack,
    /// Exceeded girth trials.
    NoMoreTrials,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAvailRows => write!(f, "no rows available"),
            Error::GirthTooSmall => write!(f, "girth is too small"),
            Error::NoMoreBacktrack => write!(f, "exceeded backtrack trials"),
            Error::NoMoreTrials => write!(f, "exceeded girth trials"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type used to indicate MacKay-Neal runtime errors.
pub type Result<T> = std::result::Result<T, Error>;

/// Configuration for the MacKay-Neal construction.
///
/// This configuration is used to set the parameters of the
/// LDPC code to construct as well as some options that affect
/// the exectution of the algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Number of rows of the parity check matrix.
    pub nrows: usize,
    /// Number of columns of the parity check matrix.
    pub ncols: usize,
    /// Maximum row weight of the parity check matrix.
    pub wr: usize,
    /// Column weight of the parity check matrix.
    pub wc: usize,
    /// Number of columns to backtrack when there are not enough
    /// available columns with weight smaller than the maximum row
    /// weight.
    pub backtrack_cols: usize,
    /// Number of times to attempt backtracking before aborting.
    pub backtrack_trials: usize,
    /// Minimum girth of the Tanner graph; `None` indicates that
    /// no constraints are imposed on the girth.
    pub min_girth: Option<usize>,
    /// Number of times to re-try generating a column to satisfy
    /// the minimum girth constraint before aborting.
    pub girth_trials: usize,
    /// Policy used to select the rows to fill.
    pub fill_policy: FillPolicy,
}

impl Config {
    /// Runs the MacKay-Neal algorith using a random seed `seed`.
    pub fn run(&self, seed: u64) -> Result<SparseMatrix> {
        MacKayNeal::new(self, seed).run()
    }

    /// Searches for a seed for a successful MacKay-Neal construction
    /// by trying several seeds.
    ///
    /// The search is performed in parallel using a parallel iterator
    /// from the rayon crate. This function returns the successful seed
    /// and the corresponding parity check matrix.
    pub fn search(&self, start_seed: u64, max_tries: u64) -> (u64, SparseMatrix) {
        (start_seed..start_seed + max_tries)
            .into_iter()
            .into_par_iter()
            .filter_map(|s| self.run(s).ok().map(|x| (s, x)))
            .find_any(|_| true)
            .expect("this should not finish if there are no successful seeds")
    }
}

/// Policy used to select the rows to fill when adding a new column
/// in the MacKay-Neal algorith.
///
/// The `Random` policy chooses rows completely randomly, and only
/// imposes the maximum row weigth constraint of the configuration.
/// This has the drawback that near the end of the algorithm too
/// many rows can be full, while other rows have too many missing
/// items. Therefore, the algorithm will fail. Even if backtracking
/// is used, it is unlikely that the algorithm suceeds when the
/// matrix is large and the LDPC code is regular, so that all the
/// rows will necessarily end up with the same row weight.
///
/// The `Uniform` policy solves this problem by always picking
/// rows which have the lower weight possible. There is still some
/// randomness involved in the selection of rows, but the only
/// choices that are considered are those with the property that
/// it is not possible to exchange one of the choosen rows by
/// another row that was not chosen and has stricly less weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillPolicy {
    /// Choose randomly from the set of rows whose weight is less
    /// than the maximum row weight.
    Random,
    /// Try to choose only among the rows which have lower weight.
    Uniform,
}

struct MacKayNeal {
    wr: usize,
    wc: usize,
    h: SparseMatrix,
    rng: Rng,
    backtrack_cols: usize,
    backtrack_trials: usize,
    min_girth: Option<usize>,
    girth_trials: usize,
    fill_policy: FillPolicy,
    current_col: usize,
}

impl MacKayNeal {
    fn new(conf: &Config, seed: u64) -> MacKayNeal {
        MacKayNeal {
            wr: conf.wr,
            wc: conf.wc,
            h: SparseMatrix::new(conf.nrows, conf.ncols),
            rng: Rng::seed_from_u64(seed),
            backtrack_cols: conf.backtrack_cols,
            backtrack_trials: conf.backtrack_trials,
            min_girth: conf.min_girth,
            girth_trials: conf.girth_trials,
            fill_policy: conf.fill_policy,
            current_col: 0,
        }
    }

    fn try_insert_column(&mut self) -> Result<()> {
        let rows = self.select_rows()?;
        self.h.insert_col(self.current_col, rows.into_iter());
        if let Some(g) = self.min_girth {
            if self
                .h
                .girth_at_node_with_max(Node::Col(self.current_col), g - 1)
                .is_some()
            {
                self.h.clear_col(self.current_col);
                return Err(Error::GirthTooSmall);
            }
        }
        Ok(())
    }

    fn select_rows(&mut self) -> Result<Vec<usize>> {
        match self.fill_policy {
            FillPolicy::Random => {
                let h = &self.h;
                let wr = self.wr;
                let avail_rows = (0..self.h.num_rows()).filter(|&r| h.row_weight(r) < wr);
                let select_rows = avail_rows.choose_multiple(&mut self.rng, self.wc);
                if select_rows.len() < self.wc {
                    return Err(Error::NoAvailRows);
                }
                Ok(select_rows)
            }
            FillPolicy::Uniform => {
                let mut avail_rows: Vec<(usize, usize)> = (0..self.h.num_rows())
                    .filter_map(|r| {
                        let w = self.h.row_weight(r);
                        if w < self.wr {
                            Some((r, w))
                        } else {
                            None
                        }
                    })
                    .collect();
                avail_rows.sort_unstable_by_key(|&(_, w)| w);
                let wc = self.wc;
                if avail_rows.len() < wc {
                    return Err(Error::NoAvailRows);
                }
                let mut sure: Vec<usize> = avail_rows
                    .iter()
                    .take_while(|&&(_, w)| w < avail_rows[wc - 1].1)
                    .map(|&(r, _)| r)
                    .collect();
                let mut additional = avail_rows
                    .iter()
                    .take_while(|&&(_, w)| w <= avail_rows[wc - 1].1)
                    .filter(|&&(_, w)| w == avail_rows[wc - 1].1)
                    .map(|&(r, _)| r)
                    .choose_multiple(&mut self.rng, wc - sure.len());
                sure.append(&mut additional);
                Ok(sure)
            }
        }
    }

    fn backtrack(&mut self) -> Result<()> {
        if self.backtrack_trials == 0 {
            return Err(Error::NoMoreBacktrack);
        }
        self.backtrack_trials -= 1;
        let b = std::cmp::min(self.current_col, self.backtrack_cols);
        let a = self.current_col - b;
        for col in a..self.current_col {
            self.h.clear_col(col);
        }
        self.current_col = a;
        Ok(())
    }

    fn retry_girth(&mut self) -> Result<()> {
        if self.girth_trials == 0 {
            return Err(Error::NoMoreTrials);
        }
        self.girth_trials -= 1;
        Ok(())
    }

    fn run(mut self) -> Result<SparseMatrix> {
        while self.current_col < self.h.num_cols() {
            match self.try_insert_column() {
                Ok(_) => self.current_col += 1,
                Err(Error::NoAvailRows) => self.backtrack()?,
                Err(Error::GirthTooSmall) => self.retry_girth()?,
                Err(e) => return Err(e),
            };
        }
        Ok(self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_matrix() {
        let conf = Config {
            nrows: 4,
            ncols: 8,
            wr: 4,
            wc: 2,
            backtrack_cols: 0,
            backtrack_trials: 0,
            min_girth: None,
            girth_trials: 0,
            fill_policy: FillPolicy::Random,
        };
        let h = conf.run(187).unwrap();
        let alist = "8 4
2 4 
2 2 2 2 2 2 2 2 
4 4 4 4 
1 3 
3 4 
1 4 
1 4 
1 2 
2 3 
2 3 
2 4 
1 3 4 5 
5 6 7 8 
1 2 6 7 
2 3 4 8 
";
        assert_eq!(h.alist(), alist);
    }
}
