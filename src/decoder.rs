//! LDPC belief propagation decoders.
//!
//! This module provides several implementations of a LDPC decoders using belief
//! propagation (the sum-product algorithm). The implementations differ in
//! details about their numerical algorithms, data types and message passing
//! schedules.

use crate::sparse::SparseMatrix;

pub mod arithmetic;
pub mod factory;
pub mod flooding;
pub mod horizontal_layered;

/// Generic LDPC decoder.
///
/// This trait is used to form LDPC decoder trait objects, abstracting over the
/// internal implementation decoder.
pub trait LdpcDecoder: std::fmt::Debug + Send {
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
    fn decode(
        &mut self,
        llrs: &[f64],
        max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput>;
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
/// This represents a message used by the flooding belief propagation
/// decoder. It is used for messages received by the nodes.
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
/// This represents a message used by the flooding belief propagation
/// decoder. It is used for messages sent by the nodes.
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

#[derive(Debug, Clone, Eq, PartialEq, Default, Hash)]
struct SentMessages<T> {
    per_source: Box<[Box<[SentMessage<T>]>]>,
}

impl<T: Default> SentMessages<T> {
    fn from_iter<I, J, B>(iter: I) -> SentMessages<T>
    where
        I: Iterator<Item = J>,
        J: Iterator<Item = B>,
        B: core::borrow::Borrow<usize>,
    {
        SentMessages {
            per_source: iter
                .map(|i| {
                    i.map(|j| SentMessage {
                        dest: *j.borrow(),
                        value: T::default(),
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    #[allow(dead_code)]
    fn send(&mut self, source: usize, destination: usize, value: T) {
        let message = self.per_source[source]
            .iter_mut()
            .find(|m| m.dest == destination)
            .expect("message for destination not found");
        message.value = value;
    }
}

fn check_llrs<T, F>(h: &SparseMatrix, llrs: &[T], hard_decision: F) -> bool
where
    T: Copy,
    F: Fn(T) -> bool,
{
    // Check if hard decision on LLRs satisfies the parity check equations
    !(0..h.num_rows()).any(|r| h.iter_row(r).filter(|&&c| hard_decision(llrs[c])).count() % 2 == 1)
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
