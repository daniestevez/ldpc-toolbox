//! LDPC decoder with horizontal layered schedule.
//!
//! This module implements a generice belief propagation LDPC decoder with a
//! serial, per check node (horizontal layered) schedule as described in [An
//! Efficient Message-Passing Schedule for LDPC
//! Decoding](https://www.eng.biu.ac.il/~goldbej/papers/engisrael.pdf), by
//! E. Sharon, S. Litsyn, and J. Goldberg.

use super::{
    DecoderOutput, LdpcDecoder, SentMessages, arithmetic::DecoderArithmetic, check_llrs,
    hard_decisions,
};
use crate::sparse::SparseMatrix;

/// LDPC belief propagation horizontal layered decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoder<A: DecoderArithmetic> {
    arithmetic: A,
    h: SparseMatrix,
    llrs: Box<[A::VarLlr]>,                        // Qv
    check_messages: SentMessages<A::CheckMessage>, // Rcv
}

impl<A: DecoderArithmetic> Decoder<A> {
    /// Creates a new horizontal layered LDPC decoder.
    ///
    /// The parameter `h` indicates the parity check matrix.
    pub fn new(h: SparseMatrix, arithmetic: A) -> Self {
        let llrs = vec![Default::default(); h.num_cols()].into_boxed_slice();
        let check_messages = SentMessages::from_iter((0..h.num_rows()).map(|r| h.iter_row(r)));
        Decoder {
            arithmetic,
            h,
            llrs,
            check_messages,
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
        assert_eq!(llrs.len(), self.llrs.len());
        let input_llrs_hard_decision = |x| x <= 0.0;
        if check_llrs(&self.h, llrs, input_llrs_hard_decision) {
            // No bit errors case
            return Ok(DecoderOutput {
                codeword: hard_decisions(llrs, input_llrs_hard_decision),
                iterations: 0,
            });
        }
        self.initialize(llrs);
        for iteration in 1..=max_iterations {
            self.process_check_nodes();
            if check_llrs(&self.h, &self.llrs, |x| {
                self.arithmetic
                    .llr_hard_decision(self.arithmetic.var_llr_to_llr(x))
            }) {
                // Decode succeeded
                return Ok(DecoderOutput {
                    codeword: hard_decisions(&self.llrs, |x| {
                        self.arithmetic
                            .llr_hard_decision(self.arithmetic.var_llr_to_llr(x))
                    }),
                    iterations: iteration,
                });
            }
        }
        // Decode failed
        Err(DecoderOutput {
            codeword: hard_decisions(&self.llrs, |x| {
                self.arithmetic
                    .llr_hard_decision(self.arithmetic.var_llr_to_llr(x))
            }),
            iterations: max_iterations,
        })
    }

    fn initialize(&mut self, llrs: &[f64]) {
        // Initialize Qv to input LLRs.
        for (x, &y) in self.llrs.iter_mut().zip(llrs.iter()) {
            *x = self
                .arithmetic
                .llr_to_var_llr(self.arithmetic.input_llr_quantize(y))
        }
        // Initialize Rcv to zero.
        for x in self.check_messages.per_source.iter_mut() {
            for msg in x.iter_mut() {
                msg.value = A::CheckMessage::default();
            }
        }
    }

    fn process_check_nodes(&mut self) {
        for messages in self.check_messages.per_source.iter_mut() {
            self.arithmetic
                .update_check_messages_and_vars(messages, &mut self.llrs);
        }
    }
}

impl<A: DecoderArithmetic> LdpcDecoder for Decoder<A> {
    fn decode(
        &mut self,
        llrs: &[f64],
        max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput> {
        Decoder::decode(self, llrs, max_iterations)
    }
}
