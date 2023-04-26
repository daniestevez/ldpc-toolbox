//! LDPC decoder factory.
//!
//! This module contains routines to build an LDPC decoder generically over the
//! arithmetic implementation. Such decoders are represented by `Box<dyn
//! LdpcDecoder>`, using the trait [`LdpcDecoder`].

use super::{
    arithmetic::{
        DecoderArithmetic, Minstarapproxf32, Minstarapproxf64, Minstarapproxi8,
        Minstarapproxi8Deg1Clip, Minstarapproxi8Jones, Minstarapproxi8JonesDeg1Clip,
        Minstarapproxi8JonesPartialHardLimit, Minstarapproxi8JonesPartialHardLimitDeg1Clip,
        Minstarapproxi8PartialHardLimit, Minstarapproxi8PartialHardLimitDeg1Clip, Phif32, Phif64,
        Tanhf32, Tanhf64,
    },
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
        llrs: &[f64],
        max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput>;
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

/// LDPC decoder implementation.
///
/// This enum lists the LDPC decoder implementations corresponding to different
/// arithmetic rules.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum DecoderImplementation {
    /// The [`Phif64`] implementation, using `f64` and the involution `phi(x)`.
    Phif64,
    /// The [`Phif32`] implementation, using `f32` and the involution `phi(x)`.
    Phif32,
    /// The [`Tanhf64`] implementation, using `f64` and the tanh rule.
    Tanhf64,
    /// The [`Tanhf32`] implementation, using `f32` and the tanh rule.
    Tanhf32,
    /// The [`Minstarapproxf64`] implementation, using `f64` and an
    /// approximation to the min* function.
    Minstarapproxf64,
    /// The [`Minstarapproxf32`] implementation, using `f32` and an
    /// approximation to the min* function.
    Minstarapproxf32,
    /// The [`Minstarapproxi8`] implementation, using 8-bit quantization and a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup).
    Minstarapproxi8,
    /// The [`Minstarapproxi8Jones`] implementation, using 8-bit quantization, a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup), and Jones clipping for variable nodes.
    Minstarapproxi8Jones,
    /// The [`Minstarapproxi8PartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and partial hard-limiting for
    /// check nodes.
    Minstarapproxi8PartialHardLimit,
    /// The [`Minstarapproxi8JonesPartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, and partial hard-limiting for check nodes.
    Minstarapproxi8JonesPartialHardLimit,
    /// The [`Minstarapproxi8Deg1Clip`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and degree-1 variable node
    /// clipping.
    Minstarapproxi8Deg1Clip,
    /// The [`Minstarapproxi8JonesDeg1Clip`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, and degree-1 variable node clipping.
    Minstarapproxi8JonesDeg1Clip,
    /// The [`Minstarapproxi8PartialHardLimitDeg1Clip`] implementation, using
    /// 8-bit quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), partial hard-limiting for check
    /// nodes, and degree-1 variable node clipping.
    Minstarapproxi8PartialHardLimitDeg1Clip,
    /// The [`Minstarapproxi8JonesPartialHardLimitDeg1Clip`] implementation,
    /// using 8-bit quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, partial hard-limiting for check nodes, and degree-1 variable node
    /// clipping.
    Minstarapproxi8JonesPartialHardLimitDeg1Clip,
}

macro_rules! impl_decoderimplementation {
    ($($var:path, $arith:ident, $text:expr);+;) => {
        impl DecoderImplementation {
            /// Builds and LDPC decoder.
            ///
            /// Given a parity check matrix, this function builds an LDPC decoder
            /// corresponding to this decoder implementation.
            pub fn build_decoder(&self, h: SparseMatrix) -> Box<dyn LdpcDecoder> {
                match self {
                    $(
                        $var => Box::new(Decoder::new(h, $arith::new())),
                    )+
                }
            }
        }

        impl std::str::FromStr for DecoderImplementation {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ok(match s {
                    $(
                        $text => $var,
                    )+
                    _ => return Err("invalid decoder implementation"),
                })
            }
        }

        impl std::fmt::Display for DecoderImplementation {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
                write!(
                    f,
                    "{}",
                    match self {
                        $(
                            $var => $text,
                        )+
                    }
                )
            }
        }
    }
}

impl_decoderimplementation!(
    DecoderImplementation::Phif64, Phif64, "Phif64";
    DecoderImplementation::Phif32, Phif32, "Phif32";
    DecoderImplementation::Tanhf64, Tanhf64, "Tanhf64";
    DecoderImplementation::Tanhf32, Tanhf32, "Tanhf32";
    DecoderImplementation::Minstarapproxf64, Minstarapproxf64, "Minstarapproxf64";
    DecoderImplementation::Minstarapproxf32, Minstarapproxf32, "Minstarapproxf32";
    DecoderImplementation::Minstarapproxi8, Minstarapproxi8, "Minstarapproxi8";
    DecoderImplementation::Minstarapproxi8Jones, Minstarapproxi8Jones, "Minstarapproxi8Jones";
    DecoderImplementation::Minstarapproxi8PartialHardLimit, Minstarapproxi8PartialHardLimit, "Minstarapproxi8PartialHardLimit";
    DecoderImplementation::Minstarapproxi8JonesPartialHardLimit, Minstarapproxi8JonesPartialHardLimit, "Minstarapproxi8JonesPartialHardLimit";
    DecoderImplementation::Minstarapproxi8Deg1Clip, Minstarapproxi8Deg1Clip, "Minstarapproxi8Deg1Clip";
    DecoderImplementation::Minstarapproxi8JonesDeg1Clip, Minstarapproxi8JonesDeg1Clip, "Minstarapproxi8JonesDeg1Clip";
    DecoderImplementation::Minstarapproxi8PartialHardLimitDeg1Clip, Minstarapproxi8PartialHardLimitDeg1Clip, "Minstarapproxi8PartialHardLimitDeg1Clip";
    DecoderImplementation::Minstarapproxi8JonesPartialHardLimitDeg1Clip, Minstarapproxi8JonesPartialHardLimitDeg1Clip, "Minstarapproxi8JonesPartialHardLimitDeg1Clip";
);
