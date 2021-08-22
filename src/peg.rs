//! # Progressive Edge Growth (PEG) LDPC construction
//!
//! This implements the algorithm described in *Xiao-Yu Hu, E. Eleftheriou and
//! D. M. Arnold, "Regular and irregular progressive edge-growth tanner graphs,"
//! in IEEE Transactions on Information Theory, vol. 51, no. 1, pp. 386-398,
//! Jan. 2005.*
//!
//! The algorithm works by adding edge by edge to the Tanner graph. For each
//! symbol node, `wc` check nodes are selected to be joined by edges. Each one
//! is selected in a different step, and the edge is added to the graph, which
//! affects subsequent decissions.
//!
//! To select an edge for the current symbol node, a breadth-first search is
//! done with that node as the root, in order to find the distance from each of
//! check nodes to the root. If there are any check nodes not yet reachable from
//! the root, a node at random is selected among the unreachable nodes that
//! have minimum degree (note that this always happens whenever the first edge
//! is added to a symbol node). If all the check nodes are rechable from the
//! root, the set of nodes of minimum degree among those nodes at maximum
//! distance from the root is selected. A node is picked at random from that
//! set.
//!
//! This procedure tries to maximize local girth greedily and to fill the
//! check nodes uniformly.

use crate::rand::{Rng, *};
use crate::sparse::{Node, SparseMatrix};
use crate::util::{compare_some, *};
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};

/// Runtime errors of the PEG construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not enought rows available.
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
/// LDPC code to construct.
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
            .sort_by_random_min(
                |(_, x, w), (_, y, v)| match compare_some(x, y).reverse() {
                    Ordering::Equal => w.cmp(v),
                    c => c,
                },
                &mut self.rng,
            )
            .ok_or(Error::NoAvailRows)?
            .0;
        self.h.insert(selected_row, col);
        Ok(())
    }

    fn run(mut self) -> Result<SparseMatrix> {
        for col in 0..self.h.num_cols() {
            for _ in 0..self.wc {
                self.insert_edge(col)?;
            }
        }
        Ok(self.h)
    }
}
