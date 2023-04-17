//! LDPC belief propagation decoder.
//!
//! This module provides several implementations of an LDPC decoder using belief
//! propagation (the sum-product algorithm). The implementations differ in
//! details about their numerical algorithms and data types.

use crate::sparse::SparseMatrix;

/// LDPC belief propagation decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoder {
    h: SparseMatrix,
    input_llrs: Box<[f64]>,
    output_llrs: Box<[f64]>,
    check_messages: Messages<f64>,
    variable_messages: Messages<f64>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
struct Message<T> {
    source: usize,
    value: T,
}

#[derive(Debug, Clone, Eq, PartialEq, Default, Hash)]
struct Messages<T> {
    per_destination: Box<[Box<[Message<T>]>]>,
}

impl Decoder {
    /// Creates a new LDPC decoder.
    ///
    /// The parameter `h` indicates the parity check matrix.
    pub fn new(h: SparseMatrix) -> Decoder {
        let input_llrs = vec![0.0; h.num_cols()].into_boxed_slice();
        let output_llrs = input_llrs.clone();
        let check_messages = Messages::from_iter((0..h.num_cols()).map(|c| h.iter_col(c)));
        let variable_messages = Messages::from_iter((0..h.num_rows()).map(|r| h.iter_row(r)));
        Decoder {
            h,
            input_llrs,
            output_llrs,
            check_messages,
            variable_messages,
        }
    }

    /// Decodes a codeword.
    ///
    /// The parameters are the LLRs for the received codeword and the maximum
    /// number of iterations to perform. If decoding is successful, the function
    /// returns the (hard decision) on the decoded codeword and the number of
    /// iterations used in decoding. If decoding is not successful, `None` is
    /// returned.
    pub fn decode(&mut self, llrs: &[f64], max_iterations: usize) -> Option<(Vec<u8>, usize)> {
        assert_eq!(llrs.len(), self.input_llrs.len());
        if self.check_llrs(llrs) {
            // No bit errors case
            return Some((Self::hard_decision(llrs), 0));
        }
        self.initialize(llrs);
        for iteration in 1..=max_iterations {
            self.process_check_nodes();
            self.process_variable_nodes();
            if self.check_llrs(&self.output_llrs) {
                // Decode succeeded
                return Some((Self::hard_decision(&self.output_llrs), iteration));
            }
        }
        // Decode failed
        None
    }

    fn initialize(&mut self, llrs: &[f64]) {
        self.input_llrs.copy_from_slice(llrs);

        // First variable messages use only input LLRs
        for (v, &llr) in self.input_llrs.iter().enumerate() {
            for &c in self.h.iter_col(v) {
                self.variable_messages.send(v, c, llr);
            }
        }
    }

    fn process_check_nodes(&mut self) {
        for (c, messages) in self.variable_messages.per_destination.iter().enumerate() {
            for (dest, value) in Self::new_check_messages(messages) {
                self.check_messages.send(c, dest, value)
            }
        }
    }

    fn new_check_messages(
        var_messages: &[Message<f64>],
    ) -> impl Iterator<Item = (usize, f64)> + '_ {
        // Compute combination of all variable messages
        let mut sign: u32 = 0;
        let mut sum = 0.0;
        let mut phis = Vec::with_capacity(var_messages.len());
        for msg in var_messages.iter() {
            let x = msg.value;
            let phi_x = phi(x.abs());
            sum += phi_x;
            phis.push(phi_x);
            if x < 0.0 {
                sign ^= 1;
            }
        }

        // Exclude the contribution of each variable to generate message for
        // that variable
        var_messages
            .iter()
            .zip(phis.into_iter())
            .map(move |(msg, phi_x)| {
                let x = msg.value;
                let y = phi(sum - phi_x);
                let s = if x < 0.0 { sign ^ 1 } else { sign };
                let val = if s == 0 { y } else { -y };
                (msg.source, val)
            })
    }

    fn process_variable_nodes(&mut self) {
        for (((v, messages), output_llr), &input_llr) in self
            .check_messages
            .per_destination
            .iter()
            .enumerate()
            .zip(self.output_llrs.iter_mut())
            .zip(self.input_llrs.iter())
        {
            let (new_llr, new_messages) = Self::new_variable_messages(input_llr, messages);
            {
                *output_llr = new_llr;
                for (dest, value) in new_messages {
                    self.variable_messages.send(v, dest, value);
                }
            }
        }
    }

    fn new_variable_messages(
        input_llr: f64,
        chk_messages: &[Message<f64>],
    ) -> (f64, impl Iterator<Item = (usize, f64)> + '_) {
        // Compute new LLR
        let llr = input_llr + chk_messages.iter().map(|m| m.value).sum::<f64>();
        // Exclude the contribution of each check node to generate message for
        // that check node
        let new_messages = chk_messages.iter().map(move |m| (m.source, llr - m.value));
        (llr, new_messages)
    }

    fn check_llrs(&self, llrs: &[f64]) -> bool {
        // Check if hard decision on LLRs satisfies the parity check equations
        !(0..self.h.num_rows()).any(|r| {
            self.h
                .iter_row(r)
                .filter(|&&c| llrs[c] <= T::zero())
                .count()
                % 2
                == 1
        })
    }

    fn hard_decision(llrs: &[f64]) -> Vec<u8> {
        llrs.iter()
            .map(|&llr| if llr <= 0.0 { 1 } else { 0 })
            .collect()
    }
}

fn phi(x: f64) -> f64 {
    // Ensure that x is not zero. Otherwise the output will be +inf, which gives
    // problems when computing (+inf) - (+inf).
    let x = x.max(1e-30);
    -((0.5 * x).tanh().ln())
}

impl<T: Default> Messages<T> {
    fn from_iter<I, J, B>(iter: I) -> Messages<T>
    where
        I: Iterator<Item = J>,
        J: Iterator<Item = B>,
        B: core::borrow::Borrow<usize>,
    {
        Messages {
            per_destination: iter
                .map(|i| {
                    i.map(|j| Message {
                        source: *j.borrow(),
                        value: T::default(),
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn send(&mut self, source: usize, destination: usize, value: T) {
        let message = self.per_destination[destination]
            .iter_mut()
            .find(|m| m.source == source)
            .expect("message for source not found");
        message.value = value;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn test_decoder() -> Decoder {
        // Example 2.5 in Sarah J. Johnson - Iterative Error Correction
        let mut h = SparseMatrix::new(4, 6);
        h.insert_row(0, [0, 1, 3].iter());
        h.insert_row(1, [1, 2, 4].iter());
        h.insert_row(2, [0, 4, 5].iter());
        h.insert_row(3, [2, 3, 5].iter());
        Decoder::new(h)
    }

    // These are based on example 2.23 in Sarah J. Johnson - Iterative Error Correction

    fn to_llrs(bits: &[u8]) -> Vec<f64> {
        bits.iter()
            .map(|&b| if b == 0 { 1.3863 } else { -1.3863 })
            .collect()
    }

    #[test]
    fn no_errors() {
        let mut decoder = test_decoder();
        let codeword = [0, 0, 1, 0, 1, 1];
        let max_iter = 100;
        let (decoded, iterations) = decoder.decode(&to_llrs(&codeword), max_iter).unwrap();
        assert_eq!(&decoded, &codeword);
        assert_eq!(iterations, 0);
    }

    #[test]
    fn single_error() {
        let mut decoder = test_decoder();
        let codeword_good = [0, 0, 1, 0, 1, 1];
        for j in 0..codeword_good.len() {
            let mut codeword_bad = codeword_good.clone();
            codeword_bad[j] ^= 1;
            let max_iter = 100;
            let (decoded, iterations) = decoder.decode(&to_llrs(&codeword_bad), max_iter).unwrap();
            assert_eq!(&decoded, &codeword_good);
            assert_eq!(iterations, 1);
        }
    }
}
