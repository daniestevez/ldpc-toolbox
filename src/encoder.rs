//! LDPC systematic encoder.
//!
//! This module implements a systematic encoder for LDPC (n, k) codes in which
//! the parity check matrix H has size (n-k) x n (i.e., has maximum rank), and
//! the square matrix formed by the last n-k columns of H is invertible. For
//! these codes, the encoder uses the first k symbols of the codeword as
//! systematic.
//!
//! The encoder works by splitting the parity check matrix as H = [H0 H1],
//! where H1 is square, and computing G0 = H1^{-1}H0. The dense matrix G0
//! is multiplied by the k message bits (as a column vector on the right) to
//! obtain the n-k parity check bits.

use crate::{gf2::GF2, sparse::SparseMatrix};
use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix1};
use num_traits::One;
use thiserror::Error;

mod gauss;

/// LDPC encoder error.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
pub enum Error {
    /// The square submatrix formed by the last columns of the parity check
    /// matrix is not invertible, so the encoder cannot be constructed.
    #[error("the square matrix formed by the last columns of the parity check is not invertible")]
    SubmatrixNotInvertible,
}

/// LDPC systematic encoder.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Encoder {
    gen_matrix: Array2<GF2>,
}

impl Encoder {
    /// Creates the systematic encoder corresponding to a parity check matrix.
    pub fn from_h(h: &SparseMatrix) -> Result<Encoder, Error> {
        let n = h.num_rows();
        let m = h.num_cols();

        // If H = [H0 H1] with H0 n x (m-n) and H1 n x n, then
        // A = [H1 H0].
        let mut a = Array2::zeros((n, m));
        for (j, k) in h.iter_all() {
            let t = if k < m - n { k + n } else { k - (m - n) };
            a[[j, t]] = GF2::one();
        }

        match gauss::gauss_reduction(&mut a) {
            Ok(()) => (),
            Err(gauss::Error::NotInvertible) => return Err(Error::SubmatrixNotInvertible),
        };

        let gen_matrix = a.slice(s![.., n..]).to_owned();
        Ok(Encoder { gen_matrix })
    }

    /// Encodes a message into a codeword.
    pub fn encode<S>(&self, message: &ArrayBase<S, Ix1>) -> Array1<GF2>
    where
        S: Data<Elem = GF2>,
    {
        let parity = self.gen_matrix.dot(message);
        ndarray::concatenate(ndarray::Axis(0), &[message.view(), parity.view()]).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num_traits::Zero;

    #[test]
    fn encode() {
        let alist = "12 4
3 9 
3 3 3 3 3 3 3 3 3 3 3 3 
9 9 9 9 
1 2 3 
1 3 4 
2 3 4 
2 3 4 
1 2 4 
1 2 3 
1 3 4 
1 2 4 
1 2 3 
2 3 4 
1 2 4 
1 3 4 
1 2 5 6 7 8 9 11 12 
1 3 4 5 6 8 9 10 11 
1 2 3 4 6 7 9 10 12 
2 3 4 5 7 8 10 11 12 
";
        let h = SparseMatrix::from_alist(alist).unwrap();
        let encoder = Encoder::from_h(&h).unwrap();
        let i = GF2::one();
        let o = GF2::zero();

        let message = [i, o, i, i, o, o, i, o];
        let codeword = encoder.encode(&ndarray::arr1(&message));
        let expected = [i, o, i, i, o, o, i, o, i, o, o, i];
        assert_eq!(&codeword.as_slice().unwrap(), &expected);

        let message = [o, i, o, o, i, i, i, o];
        let codeword = encoder.encode(&ndarray::arr1(&message));
        let expected = [o, i, o, o, i, i, i, o, i, o, i, o];
        assert_eq!(&codeword.as_slice().unwrap(), &expected);
    }
}
