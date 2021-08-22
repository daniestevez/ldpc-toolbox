//! # Progressive Edge Growth (PEG) LDPC construction

use crate::rand::{Rng, *};
use crate::sparse::{Node, SparseMatrix};
use crate::util::{compare_some, *};
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};

/// Runtime errors of the PEG construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// No rows available.
    NoAvailRows,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAvailRows => write!(f, "not enough rows available"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type used to indicate PEG runtime errors.
pub type Result<T> = std::result::Result<T, Error>;

/// Configuration for the Progressive Edge Growth construction
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
    /// Column weight of the parity check matrix.
    pub wc: usize,
}

impl Config {
    /// Runs the Progressive Edge Growth algorith using a random seed `seed`.
    pub fn run(&self, seed: u64) -> Result<SparseMatrix> {
        Peg::new(self, seed).run()
    }
}

struct Peg {
    wc: usize,
    h: SparseMatrix,
    rng: Rng,
}

impl Peg {
    fn new(conf: &Config, seed: u64) -> Peg {
        Peg {
            wc: conf.wc,
            h: SparseMatrix::new(conf.nrows, conf.ncols),
            rng: Rng::seed_from_u64(seed),
        }
    }

    fn insert_edge(&mut self, col: usize) -> Result<()> {
        let row_dist = self.h.bfs(Node::Col(col)).row_nodes_distance;
        let row_num_dist_and_weight: Vec<_> = row_dist
            .into_iter()
            .enumerate()
            .map(|(j, d)| (j, d, self.h.row_weight(j)))
            .collect();
        let selected_row = row_num_dist_and_weight
            .sort_by_random_sel(
                1,
                |(_, x, w), (_, y, v)| {
                    let c = compare_some(x, y).reverse();
                    if c == Ordering::Equal {
                        w.cmp(v)
                    } else {
                        c
                    }
                },
                &mut self.rng,
            )
            .ok_or(Error::NoAvailRows)?
            .into_iter()
            .map(|a| {
                eprintln!(
                    "PEG choosing row {} at distance {:?} with weight {}",
                    a.0, a.1, a.2
                );
                a.0
            })
            .next()
            .ok_or(Error::NoAvailRows)?;
        self.h.insert(selected_row, col);
        Ok(())
    }

    fn run(mut self) -> Result<SparseMatrix> {
        for col in 0..self.h.num_cols() {
            eprintln!("PEG for column = {}", col);
            for _ in 0..self.wc {
                self.insert_edge(col)?;
            }
        }
        Ok(self.h)
    }
}
