//! LDPC systematic encoder.
//!
//! This module implements a systematic encoder for LDPC (n, k) codes in which
//! the parity check matrix H has size (n-k) x n (i.e., has maximum rank), and
//! the square matrix formed by the last n-k columns of H is invertible. For
//! these codes, the encoder uses the first k symbols of the codeword as
//! systematic.
//!
//! There are two cases handled by the encoder, depending on the structure of
//! the parity check matrix H = [H0 H1], where H1 is square. In both cases, the
//! matrix H1 is required to be invertible.
//!
//! The first case is the case of a "staircase-type" LDPC code (this is the case
//! for DVB-S2 codes, for example). In this case, H1 has its main diagonal and
//! the diagonal below filled with ones and no other one elsewhere. An O(n)
//! encoding can be obtained by mutiplying the matrix H0 by the k message bits
//! (as a column vector on the right) and by computing the n-k running sums of
//! the components of the resulting vector of size n-k.
//!
//! In the second case (non staircase-type), the encoder computes G0 =
//! H1^{-1}H0, which in general is a dense matrix. To encode a message, the
//! matrix G0 is multiplied by the k message bits (as a column vector on the
//! right) to obtain the n-k parity check bits. In this case, the encoding
//! complexity is O(n^2).

use crate::{gf2::GF2, sparse::SparseMatrix};
use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix1};
use num_traits::One;
use thiserror::Error;

mod gauss;
mod staircase;

/// LDPC encoder error.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
pub enum Error {
    /// The square submatrix formed by the last columns of the parity check
    /// matrix is not invertible, so the encoder cannot be constructed.
    #[error("the square matrix formed by the last columns of the parity check is not invertible")]
    SubmatrixNotInvertible,
}

/// LDPC systematic encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoder {
    encoder: EncoderType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EncoderType {
    // Encoder with a general dense generator matrix for the parity.
    DenseGenerator { gen_matrix: Array2<GF2> },
    // Encoder for a staircase type (repeat-accumulate) code. The encoder sparse
    // matrix computes the parity data before accumulation.
    Staircase { gen: SparseMatrix },
}

impl Encoder {
    /// Creates the systematic encoder corresponding to a parity check matrix.
    pub fn from_h(h: &SparseMatrix) -> Result<Encoder, Error> {
        let n = h.num_rows();
        let m = h.num_cols();

        let encoder = if staircase::is_staircase(h) {
            // Special encoder for a staircase-type LDPC code.

            // If H = [H0 H1] with H0 n x (m-n) and H1 n x n, extract H0 to a
            // SparseMatrix
            let mut gen = SparseMatrix::new(n, m - n);
            for (j, k) in h.iter_all() {
                if k < m - n {
                    gen.insert(j, k);
                }
            }
            EncoderType::Staircase { gen }
        } else {
            // General case, in which the generator matrix is obtained by
            // Gaussian reduction (it will be a dense matrix in general).

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
            EncoderType::DenseGenerator { gen_matrix }
        };
        Ok(Encoder { encoder })
    }

    /// Encodes a message into a codeword.
    pub fn encode<S>(&self, message: &ArrayBase<S, Ix1>) -> Array1<GF2>
    where
        S: Data<Elem = GF2>,
    {
        let parity = match &self.encoder {
            EncoderType::DenseGenerator { gen_matrix } => gen_matrix.dot(message),
            EncoderType::Staircase { gen } => {
                // initial parity (needs to be accumulated)
                let mut parity = Array1::from_iter(
                    (0..gen.num_rows()).map(|j| gen.iter_row(j).map(|&k| message[k]).sum()),
                );
                // Accumulate parity
                for j in 1..parity.len() {
                    let previous = parity[j - 1];
                    parity[j] += previous;
                }
                parity
            }
        };
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

    #[test]
    fn encode_staircase() {
        let alist = "5 3
2 4
2 2 2 2 1
2 4 4
1 3
2 3
1 2
2 3
3
1 3
2 3 4
1 2 4 5
";
        let h = SparseMatrix::from_alist(alist).unwrap();
        let encoder = Encoder::from_h(&h).unwrap();
        assert!(matches!(encoder.encoder, EncoderType::Staircase { .. }));
        let i = GF2::one();
        let o = GF2::zero();

        let message = [i, o];
        let codeword = encoder.encode(&ndarray::arr1(&message));
        let expected = [i, o, i, i, o];
        assert_eq!(&codeword.as_slice().unwrap(), &expected);

        let message = [o, i];
        let codeword = encoder.encode(&ndarray::arr1(&message));
        let expected = [o, i, o, i, o];
        assert_eq!(&codeword.as_slice().unwrap(), &expected);
    }
}
