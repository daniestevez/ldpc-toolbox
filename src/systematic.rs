//! Systematic code constructions.
//!
//! This module contains a function [`parity_to_systematic`] that can be used to
//! convert a full-rank parity check matrix into one that supports systematic
//! encoding using the first variables (as done by the systematic encoder in the
//! [`encoder`](crate::encoder) module) by permuting the columns of the parity
//! check matrix.

use crate::{gf2::GF2, linalg, sparse::SparseMatrix};
use ndarray::Array2;
use num_traits::{One, Zero};
use thiserror::Error;

/// Systematic construction error.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
pub enum Error {
    /// The parity check matrix has more rows than columns.
    #[error("the parity check matrix has more rows than columns")]
    ParityOverdetermined,
    /// The parity check matrix does not have full rank.
    #[error("the parity check matrix does not have full rank")]
    NotFullRank,
}

/// Permutes the columns of the parity check matrix to obtain a parity check
/// matrix that supports systematic encoding using the first variables.
///
/// This function returns a parity check matrix obtaining by permuting the
/// columns of `h` in such a way that the square submatrix formed by the last
/// columns is invertible.
pub fn parity_to_systematic(h: &SparseMatrix) -> Result<SparseMatrix, Error> {
    let n = h.num_rows();
    let m = h.num_cols();
    if n > m {
        return Err(Error::ParityOverdetermined);
    }
    let mut a = Array2::zeros((n, m));
    for (j, k) in h.iter_all() {
        a[[j, k]] = GF2::one();
    }
    linalg::row_echelon_form(&mut a);
    // Check that the matrix has full rank by checking that there is a non-zero
    // element in the last row (we start looking by the end, since chances are
    // higher to find a non-zero element there).
    if !(0..m).rev().any(|j| a[[n - 1, j]] != GF2::zero()) {
        return Err(Error::NotFullRank);
    }
    // write point for columns that do not "go down" in the row echelon form
    let mut k = 0;
    let mut h_new = SparseMatrix::new(n, m);
    let mut j0 = 0;
    for j in 0..n {
        assert!(k < m - n);
        let mut found = false;
        for s in j0..m {
            if a[[j, s]] == GF2::zero() {
                // Column does not "go down" on row echelon form. Place it at the current write point.
                for &u in h.iter_col(s) {
                    h_new.insert(u, k);
                }
                k += 1;
            } else {
                // Column goes down on row echelon form. Move to its appropriate
                // position in the last columns.
                let col = m - n + j;
                for &u in h.iter_col(s) {
                    h_new.insert(u, col);
                }
                found = true;
                j0 = s + 1;
                break;
            }
        }
        assert!(found);
    }
    // Insert remaining columns at the write point
    for j in j0..m {
        assert!(k < m - n);
        for &u in h.iter_col(j) {
            h_new.insert(u, k);
        }
        k += 1;
    }
    Ok(h_new)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn to_systematic() {
        let mut h = SparseMatrix::new(3, 9);
        h.insert_col(0, [0, 1, 2].into_iter());
        h.insert_col(1, [0, 2].into_iter());
        // h.insert_col(2, [].into_iter()); this does nothing and does not compile
        h.insert_col(3, [1].into_iter());
        h.insert_col(4, [0, 1].into_iter());
        h.insert_col(5, [1, 2].into_iter());
        h.insert_col(6, [0, 2].into_iter());
        h.insert_col(7, [1].into_iter());
        h.insert_col(8, [0, 2].into_iter());
        let mut expected = SparseMatrix::new(3, 9);
        expected.insert_col(6, [0, 1, 2].into_iter());
        expected.insert_col(7, [0, 2].into_iter());
        // expected.insert_col(0, [].into_iter()); this does nothing and does not compile
        expected.insert_col(1, [1].into_iter());
        expected.insert_col(8, [0, 1].into_iter());
        expected.insert_col(2, [1, 2].into_iter());
        expected.insert_col(3, [0, 2].into_iter());
        expected.insert_col(4, [1].into_iter());
        expected.insert_col(5, [0, 2].into_iter());
        assert_eq!(parity_to_systematic(&h).unwrap(), expected);
    }
}
