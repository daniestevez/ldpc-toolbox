//! Code puncturing.

use ndarray::{s, Array1, ArrayBase, Data, Ix1};
use thiserror::Error;

/// Puncturer.
///
/// This struct is used to perform puncturing on codewords to be transmitted,
/// and "depuncturing" on demodulated LLRs.
#[derive(Debug, Clone)]
pub struct Puncturer {
    pattern: Box<[bool]>,
    num_trues: usize,
}

/// Puncturer error.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Error)]
pub enum Error {
    /// The codeword size is not divisible by the puncturing pattern length
    #[error("codeword size not divisible by puncturing pattern length")]
    CodewordSizeNotDivisible,
}

impl Puncturer {
    /// Creates a new puncturer.
    ///
    /// The puncturing pattern is defined by blocks. For example `[true, true,
    /// true, false]` means that the first 3/4 of the codeword bits are
    /// preserved, and the last 1/4 is punctured.
    ///
    /// # Panics
    ///
    /// This function panics if the pattern is empty.
    pub fn new(pattern: &[bool]) -> Puncturer {
        assert!(!pattern.is_empty());
        Puncturer {
            pattern: pattern.into(),
            num_trues: pattern.iter().filter(|&&b| b).count(),
        }
    }

    /// Puncture a codeword.
    ///
    /// Given a codeword, returns the punctured codeword. An error is returned
    /// if the length of the codeword is not divisible by the length of the
    /// puncturing pattern.
    pub fn puncture<S, A>(&self, codeword: &ArrayBase<S, Ix1>) -> Result<Array1<A>, Error>
    where
        S: Data<Elem = A>,
        A: Clone,
    {
        let pattern_len = self.pattern.len();
        let codeword_len = codeword.shape()[0];
        if codeword_len % pattern_len != 0 {
            return Err(Error::CodewordSizeNotDivisible);
        }
        let block_size = codeword_len / pattern_len;
        let output_size = block_size * self.num_trues;
        let mut out = Array1::uninit(output_size);
        for (j, k) in self
            .pattern
            .iter()
            .enumerate()
            .filter_map(|(k, &b)| if b { Some(k) } else { None })
            .enumerate()
        {
            codeword
                .slice(s![k * block_size..(k + 1) * block_size])
                .assign_to(out.slice_mut(s![j * block_size..(j + 1) * block_size]));
        }
        // Safety: all the elements of out have been assigned by the loop above.
        Ok(unsafe { out.assume_init() })
    }

    /// Depuncture LLRs.
    ///
    /// This function depunctures demodulated LLRs by inserting zeros (which
    /// indicate erasures) in the positions of the codeword that were
    /// punctured. The input length must correspond to the punctured codeword,
    /// while the output length is equal to the codeword length. An error is
    /// returned if the length of input is not divisible by the number of `true`
    /// elements in the pattern.
    pub fn depuncture(&self, llrs: &[f32]) -> Result<Vec<f32>, Error> {
        if llrs.len() % self.num_trues != 0 {
            return Err(Error::CodewordSizeNotDivisible);
        }
        let block_size = llrs.len() / self.num_trues;
        let output_size = self.pattern.len() * block_size;
        let mut output = vec![0.0; output_size];
        for (j, k) in self
            .pattern
            .iter()
            .enumerate()
            .filter_map(|(k, &b)| if b { Some(k) } else { None })
            .enumerate()
        {
            output[k * block_size..(k + 1) * block_size]
                .copy_from_slice(&llrs[j * block_size..(j + 1) * block_size]);
        }
        Ok(output)
    }

    /// Returns the rate of the puncturer.
    ///
    /// The rate is defined as the length of the original codeword divided by
    /// the length of the punctured codeword, and so it is always greater or
    /// equal to one.
    pub fn rate(&self) -> f64 {
        self.pattern.len() as f64 / self.num_trues as f64
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn puncturing() {
        let puncturer = Puncturer::new(&[true, true, false, true, false]);
        let codeword = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let punctured = puncturer.puncture(&codeword).unwrap();
        let expected = array![0, 1, 2, 3, 6, 7];
        assert_eq!(&punctured, &expected);
        let llrs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let llrs_out = puncturer.depuncture(&llrs).unwrap();
        let expected = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0];
        assert_eq!(&llrs_out, &expected);
    }
}
