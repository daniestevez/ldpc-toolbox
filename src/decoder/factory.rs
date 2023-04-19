//! LDPC decoder factory.
//!
//! This module contains routines to build an LDPC decoder generically over the
//! arithmetic implementation. Such decoders are represented by `Box<dyn
//! LdpcDecoder>`, using the trait [`LdpcDecoder`].

use super::{
    arithmetic::{DecoderArithmetic, Phif64},
    Decoder, DecoderOutput,
};
use crate::sparse::SparseMatrix;

/// Generic LDPC decoder.
///
/// This trait is used to form LDPC decoder trait objects, abstracting over the
/// implementation of the decoder arithmetic.
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
        llrs: &[f32],
        max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput>;
}

impl<A: DecoderArithmetic> LdpcDecoder for Decoder<A> {
    fn decode(
        &mut self,
        llrs: &[f32],
        max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput> {
        Decoder::decode(self, llrs, max_iterations)
    }
}

/// LDPC decoder implementation.
///
/// This enum lists the LDPC decoder implementations corresponding to different
/// arithmetic rules.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum DecoderImplementation {
    /// The [`Phif64`] implementation, using `f64` and the involution `phi(x)`.
    Phif64,
}

impl DecoderImplementation {
    /// Builds and LDPC decoder.
    ///
    /// Given a parity check matrix, this function builds an LDPC decoder
    /// corresponding to this decoder implementation.
    pub fn build_decoder(&self, h: SparseMatrix) -> Box<dyn LdpcDecoder> {
        match self {
            DecoderImplementation::Phif64 => Box::new(Decoder::new(h, Phif64::new())),
        }
    }
}

impl std::str::FromStr for DecoderImplementation {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "Phif64" => DecoderImplementation::Phif64,
            _ => return Err("invalid decoder implementation"),
        })
    }
}
