//! Codeword bit interleaving.
//!
//! This module implements an interleaver and deinterleaver. The interleaver can
//! be used before modulating the codeword into symbols. The implementation is
//! compliant with the DVB-S2 interleaver.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1};
use num_traits::Zero;

/// Interleaver.
///
/// Implements matrix interleaving/deinterleaving of codeword bits following the
/// DVB-S2 standard.
#[derive(Debug, Clone)]
pub struct Interleaver {
    columns: usize,
    read_rows_backwards: bool,
}

impl Interleaver {
    /// Creates a new interleaver.
    ///
    /// The `columns` parameter defines the number of columns of the
    /// interleaver.Typically this is the number of bits per symbol.
    ///
    /// The `read_rows_backwards` parameter controls whether the rows should be
    /// read backwards. In DVB-S2 this option is only used in 8PSK rate 3/5.
    pub fn new(columns: usize, read_rows_backwards: bool) -> Interleaver {
        Interleaver {
            columns,
            read_rows_backwards,
        }
    }

    /// Interleaves a codeword.
    ///
    /// # Panics
    ///
    /// Panics if the codeword size is not divisible by the number of columns.
    pub fn interleave<S, T: Clone + Zero>(&self, codeword: &ArrayBase<S, Ix1>) -> Array1<T>
    where
        S: Data<Elem = T>,
    {
        assert_eq!(codeword.len() % self.columns, 0);
        let a2 = codeword
            .view()
            .into_shape((self.columns, codeword.len() / self.columns))
            .unwrap();
        let mut transpose = a2.t();
        if self.read_rows_backwards {
            transpose.invert_axis(Axis(1));
        }
        let mut a = Array2::zeros(transpose.raw_dim());
        a.assign(&transpose);
        a.view().into_shape(codeword.len()).unwrap().to_owned()
    }

    /// Deinterleaves a codeword.
    ///
    /// # Panics
    ///
    /// Panics if the codeword size is not divisible by the number of columns.
    pub fn deinterleave<T: Clone + Zero>(&self, codeword: &[T]) -> Vec<T> {
        assert_eq!(codeword.len() % self.columns, 0);
        let codeword = Array1::from_iter(codeword.iter().cloned());
        let a2 = codeword
            .view()
            .into_shape((codeword.len() / self.columns, self.columns))
            .unwrap();
        let mut transpose = a2.t();
        if self.read_rows_backwards {
            transpose.invert_axis(Axis(0));
        }
        let mut a = Array2::zeros(transpose.raw_dim());
        a.assign(&transpose);
        a.view()
            .into_shape(codeword.len())
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn interleaver_3() {
        let interleaver = Interleaver::new(3, false);
        let interleaved = interleaver.interleave(&ndarray::arr1(&[0, 1, 2, 3, 4, 5]));
        let expected = [0, 2, 4, 1, 3, 5];
        assert_eq!(interleaved.as_slice().unwrap(), &expected);
    }

    #[test]
    fn interleaver_3_backwards() {
        let interleaver = Interleaver::new(3, true);
        let interleaved = interleaver.interleave(&ndarray::arr1(&[0, 1, 2, 3, 4, 5]));
        let expected = [4, 2, 0, 5, 3, 1];
        assert_eq!(interleaved.as_slice().unwrap(), &expected);
    }

    #[test]
    fn deinterleaver_3() {
        let interleaver = Interleaver::new(3, false);
        let original = [0, 1, 2, 3, 4, 5];
        let interleaved = interleaver.interleave(&ndarray::arr1(&original));
        let deinterleaved = interleaver.deinterleave(interleaved.as_slice().unwrap());
        assert_eq!(&deinterleaved, &original);
    }

    #[test]
    fn deinterleaver_3_backwards() {
        let interleaver = Interleaver::new(3, true);
        let original = [0, 1, 2, 3, 4, 5];
        let interleaved = interleaver.interleave(&ndarray::arr1(&original));
        let deinterleaved = interleaver.deinterleave(interleaved.as_slice().unwrap());
        assert_eq!(&deinterleaved, &original);
    }
}
