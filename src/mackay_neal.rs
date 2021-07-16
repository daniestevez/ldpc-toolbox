//! # MacKay-Neal pseudorandom LDPC construction
//!
//! This implements the algorithms from *MacKay, D.J. and Neal, R.M., 1996.
//! Near Shannon limit performance of low density parity check codes.
//! Electronics letters, 32(18), p.1645.* and variations on this idea.

use crate::rand::{Rng, *};
use crate::sparse::SparseMatrix;
use rand::seq::IteratorRandom;

/// A [`String`] with an description of the error.
pub type Error = String;
/// A [`Result`] type containing an error [`String`].
pub type Result<T> = std::result::Result<T, Error>;

/// The simplest MacKay-Neal algorithm possible. It uses constant weights
/// and does not check for the introduction of short cycles.
///
/// The algorithm works by adding column by column to the parity check
/// matrix. At each step, `wc` rows from the subset of rows that have not yet
/// achieved the total row weight `wr` are random chosen, and ones are inserted
/// in those positions.
///
/// The random `seed` is used to obtain repeatable results.
///
/// # Errors
/// If the column weight cannot be satisfied at some point due to no available
/// rows, an error is returned.
///
/// # Examples
/// ```
/// # use ldpc_toolbox::mackay_neal::simple;
/// let seed = 42;
/// let h = simple(4, 8, 4, 2, seed).unwrap();
/// println!("{}", h.alist());
/// ```
pub fn simple(nrows: usize, ncols: usize, wr: usize, wc: usize, seed: u64) -> Result<SparseMatrix> {
    simple_min_girth(nrows, ncols, wr, wc, None, 1, seed)
}

/// A simple MacKay-Neal algorithm with minimum girth setting.
///
/// This algorithm works like `simple()` but at each step the candidate
/// column is checked to see if it maintains the girth of the graph at or above
/// some minimum `min_girth`. If not, another random candidate column is
/// chosen according to the available rows. The algorithm aborts after
/// `trials` random choices that are unable to yield a new column satisfying
/// the required properties.
pub fn simple_min_girth(
    nrows: usize,
    ncols: usize,
    wr: usize,
    wc: usize,
    min_girth: Option<usize>,
    trials: usize,
    seed: u64,
) -> Result<SparseMatrix> {
    let mut h = SparseMatrix::new(nrows, ncols);
    let mut rng = Rng::seed_from_u64(seed);
    for col in 0..ncols {
        let mut correct = false;
        for _ in 0..trials {
            let avail_rows = (0..nrows).filter(|r| h.row_weight(*r) < wr);
            let select_rows = avail_rows.choose_multiple(&mut rng, wc);
            if select_rows.len() < wc {
                return Err(String::from(
                    "not enough available rows to satisfy column weight",
                ));
            }
            h.set_col(col, select_rows.into_iter());
            if let Some(g) = min_girth {
                if h.girth_at_col_with_max(col, g - 1).is_none() {
                    correct = true;
                    break;
                }
            } else {
                correct = true;
                break;
            }
        }
        if !correct {
            return Err(String::from(
                "exceeded iterations to find a column with acceptable girth",
            ));
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() {
        let h = simple(4, 8, 4, 2, 187).unwrap();
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
