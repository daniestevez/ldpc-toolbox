//! LDPC belief propagation decoder.
//!
//! This module provides several implementations of an LDPC decoder using belief
//! propagation (the sum-product algorithm). The implementations differ in
//! details about their numerical algorithms and data types.

use crate::sparse::SparseMatrix;

pub mod arithmetic;
use arithmetic::DecoderArithmetic;

pub mod factory;

/// LDPC belief propagation decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoder<A: DecoderArithmetic> {
    arithmetic: A,
    h: SparseMatrix,
    input_llrs: Box<[A::Llr]>,
    output_llrs: Box<[A::Llr]>,
    check_messages: Messages<A::CheckMessage>,
    variable_messages: Messages<A::VarMessage>,
}

/// LDPC decoder output.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct DecoderOutput {
    /// Decoded codeword.
    ///
    /// Contains the hard decision bits of the decoded codeword.
    pub codeword: Vec<u8>,
    /// Number of iterations.
    ///
    /// Number of iterations used in decoding.
    pub iterations: usize,
}

/// LDPC decoder message.
///
/// This represents a message used by the belief propagation decoder. It is used
/// for messages received by the nodes.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
pub struct Message<T> {
    /// Message source.
    ///
    /// Contains the index of the variable node or check node that sent the
    /// message.
    pub source: usize,
    /// Value.
    ///
    /// Contains the value of the message.
    pub value: T,
}

/// LDPC decoder outgoing message.
///
/// This represents a message used by the belief propagation decoder. It is used
/// for messages sent by the nodes.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
pub struct SentMessage<T> {
    /// Message destination.
    ///
    /// Contains the index of the variable node or check node to which the
    /// message is addressed.
    pub dest: usize,
    /// Value.
    ///
    /// Contains the value of the message.
    pub value: T,
}

#[derive(Debug, Clone, Eq, PartialEq, Default, Hash)]
struct Messages<T> {
    per_destination: Box<[Box<[Message<T>]>]>,
}

impl<A: DecoderArithmetic> Decoder<A> {
    /// Creates a new LDPC decoder.
    ///
    /// The parameter `h` indicates the parity check matrix.
    pub fn new(h: SparseMatrix, arithmetic: A) -> Self {
        let input_llrs = vec![Default::default(); h.num_cols()].into_boxed_slice();
        let output_llrs = input_llrs.clone();
        let check_messages = Messages::from_iter((0..h.num_cols()).map(|c| h.iter_col(c)));
        let variable_messages = Messages::from_iter((0..h.num_rows()).map(|r| h.iter_row(r)));
        Decoder {
            arithmetic,
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
    /// returns an `Ok` containing the (hard decision) on the decoded codeword
    /// and the number of iterations used in decoding. If decoding is not
    /// successful, the function returns an `Err` containing the hard decision
    /// on the final decoder LLRs (which still has some bit errors) and the
    /// number of iterations used in decoding (which is equal to
    /// `max_iterations`).
    pub fn decode(
        &mut self,
        llrs: &[f64],
        max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput> {
        assert_eq!(llrs.len(), self.input_llrs.len());
        let input_llrs_hard_decision = |x| x <= 0.0;
        if self.check_llrs(llrs, input_llrs_hard_decision) {
            // No bit errors case
            return Ok(DecoderOutput {
                codeword: Self::hard_decisions(llrs, input_llrs_hard_decision),
                iterations: 0,
            });
        }
        self.initialize(llrs);
        for iteration in 1..=max_iterations {
            self.process_check_nodes();
            self.process_variable_nodes();
            if self.check_llrs(&self.output_llrs, |x| self.arithmetic.llr_hard_decision(x)) {
                // Decode succeeded
                return Ok(DecoderOutput {
                    codeword: Self::hard_decisions(&self.output_llrs, |x| {
                        self.arithmetic.llr_hard_decision(x)
                    }),
                    iterations: iteration,
                });
            }
        }
        // Decode failed
        Err(DecoderOutput {
            codeword: Self::hard_decisions(&self.output_llrs, |x| {
                self.arithmetic.llr_hard_decision(x)
            }),
            iterations: max_iterations,
        })
    }

    fn initialize(&mut self, llrs: &[f64]) {
        for (x, &y) in self.input_llrs.iter_mut().zip(llrs.iter()) {
            *x = self.arithmetic.input_llr_quantize(y)
        }

        // First variable messages use only input LLRs
        for (v, &llr) in self.input_llrs.iter().enumerate() {
            for &c in self.h.iter_col(v) {
                self.variable_messages
                    .send(v, c, self.arithmetic.llr_to_var_message(llr));
            }
        }
    }

    fn process_check_nodes(&mut self) {
        for (c, messages) in self.variable_messages.per_destination.iter().enumerate() {
            self.arithmetic.send_check_messages(messages, {
                let check_messages = &mut self.check_messages;
                move |msg| check_messages.send(c, msg.dest, msg.value)
            });
        }
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
            *output_llr = self.arithmetic.send_var_messages(input_llr, messages, {
                let var_messages = &mut self.variable_messages;
                move |msg| var_messages.send(v, msg.dest, msg.value)
            });
        }
    }

    fn check_llrs<T, F>(&self, llrs: &[T], hard_decision: F) -> bool
    where
        T: Copy,
        F: Fn(T) -> bool,
    {
        // Check if hard decision on LLRs satisfies the parity check equations
        !(0..self.h.num_rows()).any(|r| {
            self.h
                .iter_row(r)
                .filter(|&&c| hard_decision(llrs[c]))
                .count()
                % 2
                == 1
        })
    }

    fn hard_decisions<T, F>(llrs: &[T], hard_decision: F) -> Vec<u8>
    where
        T: Copy,
        F: Fn(T) -> bool,
    {
        llrs.iter()
            .map(|&llr| u8::from(hard_decision(llr)))
            .collect()
    }
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
    use super::arithmetic::Phif64;
    use super::*;

    fn test_decoder() -> Decoder<Phif64> {
        // Example 2.5 in Sarah J. Johnson - Iterative Error Correction
        let mut h = SparseMatrix::new(4, 6);
        h.insert_row(0, [0, 1, 3].iter());
        h.insert_row(1, [1, 2, 4].iter());
        h.insert_row(2, [0, 4, 5].iter());
        h.insert_row(3, [2, 3, 5].iter());
        Decoder::new(h, Phif64::new())
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
        let DecoderOutput {
            codeword: decoded,
            iterations,
        } = decoder.decode(&to_llrs(&codeword), max_iter).unwrap();
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
            let DecoderOutput {
                codeword: decoded,
                iterations,
            } = decoder.decode(&to_llrs(&codeword_bad), max_iter).unwrap();
            assert_eq!(&decoded, &codeword_good);
            assert_eq!(iterations, 1);
        }
    }
}
