//! LDPC decoder factory.
//!
//! This module contains routines to build an LDPC decoder generically over
//! different internal implementations. Such decoders are represented by
//! `Box<dyn LdpcDecoder>`, using the trait [`LdpcDecoder`].

use super::{arithmetic::*, flooding, horizontal_layered, LdpcDecoder};
use crate::sparse::SparseMatrix;
use std::fmt::Display;

/// Decoder factory.
///
/// This trait is implemented by [`DecoderImplementation`], which builds a
/// suitable decoder depending on the value of an enum. Other factories can be
/// implemented by the user in order to run a BER test with an LDPC decoder
/// implemented externally to ldpc-toolbox (such decoder must be wrapped as a
/// `Box <dyn LdpcDecoder>`).
pub trait DecoderFactory: Display + Clone + Sync + Send + 'static {
    /// Builds and LDPC decoder.
    ///
    /// Given a parity check matrix, this function builds an LDPC decoder
    /// corresponding to this decoder implementation.
    fn build_decoder(&self, h: SparseMatrix) -> Box<dyn LdpcDecoder>;
}

/// LDPC decoder implementation.
///
/// This enum lists the LDPC decoder implementations corresponding to different
/// arithmetic rules.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum DecoderImplementation {
    /// The [`Phif64`] implementation, using `f64` and the involution
    /// `phi(x)`. This uses a flooding schedule.
    Phif64,
    /// The [`Phif32`] implementation, using `f32` and the involution
    /// `phi(x)`. This uses a flooding schedule.
    Phif32,
    /// The [`Tanhf64`] implementation, using `f64` and the tanh rule. This uses
    /// a flooding schedule.
    Tanhf64,
    /// The [`Tanhf32`] implementation, using `f32` and the tanh rule. This uses
    /// a flooding schedule.
    Tanhf32,
    /// The [`Minstarapproxf64`] implementation, using `f64` and an
    /// approximation to the min* function. This uses a flooding schedule.
    Minstarapproxf64,
    /// The [`Minstarapproxf32`] implementation, using `f32` and an
    /// approximation to the min* function. This uses a flooding schedule.
    Minstarapproxf32,
    /// The [`Minstarapproxi8`] implementation, using 8-bit quantization and a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup). This uses a flooding schedule.
    Minstarapproxi8,
    /// The [`Minstarapproxi8Jones`] implementation, using 8-bit quantization, a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup), and Jones clipping for variable nodes. This uses a
    /// flooding schedule.
    Minstarapproxi8Jones,
    /// The [`Minstarapproxi8PartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and partial hard-limiting for
    /// check nodes. This uses a flooding schedule.
    Minstarapproxi8PartialHardLimit,
    /// The [`Minstarapproxi8JonesPartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, and partial hard-limiting for check nodes. This uses a flooding
    /// schedule.
    Minstarapproxi8JonesPartialHardLimit,
    /// The [`Minstarapproxi8Deg1Clip`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and degree-1 variable node
    /// clipping. This uses a flooding schedule.
    Minstarapproxi8Deg1Clip,
    /// The [`Minstarapproxi8JonesDeg1Clip`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, and degree-1 variable node clipping. This uses a flooding
    /// schedule.
    Minstarapproxi8JonesDeg1Clip,
    /// The [`Minstarapproxi8PartialHardLimitDeg1Clip`] implementation, using
    /// 8-bit quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), partial hard-limiting for check
    /// nodes, and degree-1 variable node clipping. This uses a flooding
    /// schedule.
    Minstarapproxi8PartialHardLimitDeg1Clip,
    /// The [`Minstarapproxi8JonesPartialHardLimitDeg1Clip`] implementation,
    /// using 8-bit quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, partial hard-limiting for check nodes, and degree-1 variable node
    /// clipping. This uses a flooding schedule.
    Minstarapproxi8JonesPartialHardLimitDeg1Clip,
    /// The [`Aminstarf64`] implementation, using `f64` and an approximation to
    /// the min* function. This uses a flooding schedule.
    Aminstarf64,
    /// The [`Aminstarf32`] implementation, using `f32` and an approximation to
    /// the min* function. This uses a flooding schedule.
    Aminstarf32,
    /// The [`Aminstari8`] implementation, using 8-bit quantization and a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup). This uses a flooding schedule.
    Aminstari8,
    /// The [`Aminstari8Jones`] implementation, using 8-bit quantization, a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup), and Jones clipping for variable nodes. This uses a
    /// flooding schedule.
    Aminstari8Jones,
    /// The [`Aminstari8PartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and partial hard-limiting for
    /// check nodes. This uses a flooding schedule.
    Aminstari8PartialHardLimit,
    /// The [`Aminstari8JonesPartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, and partial hard-limiting for check nodes. This uses a flooding
    /// schedule.
    Aminstari8JonesPartialHardLimit,
    /// The [`Aminstari8Deg1Clip`] implementation, using 8-bit quantization, a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup), and degree-1 variable node clipping. This uses a flooding
    /// schedule.
    Aminstari8Deg1Clip,
    /// The [`Aminstari8JonesDeg1Clip`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, and degree-1 variable node clipping. This uses a flooding
    /// schedule.
    Aminstari8JonesDeg1Clip,
    /// The [`Aminstari8PartialHardLimitDeg1Clip`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), partial hard-limiting for check
    /// nodes, and degree-1 variable node clipping. This uses a flooding
    /// schedule.
    Aminstari8PartialHardLimitDeg1Clip,
    /// The [`Aminstari8JonesPartialHardLimitDeg1Clip`] implementation, using
    /// 8-bit quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), Jones clipping for variable
    /// nodes, partial hard-limiting for check nodes, and degree-1 variable node
    /// clipping. This uses a flooding schedule.
    Aminstari8JonesPartialHardLimitDeg1Clip,
    /// The [`Phif64`] implementation, using `f64` and the involution
    /// `phi(x)`. This uses a horizontal layered schedule.
    HLPhif64,
    /// The [`Phif32`] implementation, using `f32` and the involution
    /// `phi(x)`. This uses a horizontal layered schedule.
    HLPhif32,
    /// The [`Tanhf64`] implementation, using `f64` and the tanh rule. This uses
    /// a horizontal layered schedule.
    HLTanhf64,
    /// The [`Tanhf32`] implementation, using `f32` and the tanh rule. This uses
    /// a horizontal layered schedule.
    HLTanhf32,
    /// The [`Minstarapproxf64`] implementation, using `f64` and an
    /// approximation to the min* function. This uses a horizontal layered
    /// schedule.
    HLMinstarapproxf64,
    /// The [`Minstarapproxf32`] implementation, using `f32` and an
    /// approximation to the min* function. This uses a horizontal layered
    /// schedule.
    HLMinstarapproxf32,
    /// The [`Minstarapproxi8`] implementation, using 8-bit quantization and a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup). This uses a horizontal layered schedule.
    HLMinstarapproxi8,
    /// The [`Minstarapproxi8PartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and partial hard-limiting for
    /// check nodes. This uses a horizontal layered schedule.
    HLMinstarapproxi8PartialHardLimit,
    /// The [`Aminstarf64`] implementation, using `f64` and an approximation to
    /// the min* function. This uses a horizontal layered schedule.
    HLAminstarf64,
    /// The [`Aminstarf32`] implementation, using `f32` and an approximation to
    /// the min* function. This uses a horizontal layered schedule.
    HLAminstarf32,
    /// The [`Aminstari8`] implementation, using 8-bit quantization and a
    /// quantized approximation to the min* function (implemented using small
    /// table lookup). This uses a horizontal layered schedule.
    HLAminstari8,
    /// The [`Aminstari8PartialHardLimit`] implementation, using 8-bit
    /// quantization, a quantized approximation to the min* function
    /// (implemented using small table lookup), and partial hard-limiting for
    /// check nodes. This uses a horizontal layered schedule.
    HLAminstari8PartialHardLimit,
}

macro_rules! new_decoder {
    (flooding, $arith:ty, $h:expr) => {
        flooding::Decoder::new($h, <$arith>::new())
    };
    (horizontal_layered, $arith:ty, $h:expr) => {
        horizontal_layered::Decoder::new($h, <$arith>::new())
    };
}

macro_rules! impl_decoderimplementation {
    ($($var:path, $arith:ty, $decoder:tt, $text:expr);+;) => {
        impl DecoderFactory for DecoderImplementation {
            fn build_decoder(&self, h: SparseMatrix) -> Box<dyn LdpcDecoder> {
                match self {
                    $(
                        $var => Box::new(new_decoder!($decoder, $arith, h)),
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

        impl Display for DecoderImplementation {
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
    DecoderImplementation::Phif64, Phif64, flooding, "Phif64";
    DecoderImplementation::Phif32, Phif32, flooding, "Phif32";
    DecoderImplementation::Tanhf64, Tanhf64, flooding, "Tanhf64";
    DecoderImplementation::Tanhf32, Tanhf32, flooding, "Tanhf32";
    DecoderImplementation::Minstarapproxf64, Minstarapproxf64, flooding, "Minstarapproxf64";
    DecoderImplementation::Minstarapproxf32, Minstarapproxf32, flooding, "Minstarapproxf32";
    DecoderImplementation::Minstarapproxi8, Minstarapproxi8, flooding, "Minstarapproxi8";
    DecoderImplementation::Minstarapproxi8Jones, Minstarapproxi8Jones, flooding, "Minstarapproxi8Jones";
    DecoderImplementation::Minstarapproxi8PartialHardLimit, Minstarapproxi8PartialHardLimit, flooding, "Minstarapproxi8PartialHardLimit";
    DecoderImplementation::Minstarapproxi8JonesPartialHardLimit, Minstarapproxi8JonesPartialHardLimit, flooding, "Minstarapproxi8JonesPartialHardLimit";
    DecoderImplementation::Minstarapproxi8Deg1Clip, Minstarapproxi8Deg1Clip, flooding, "Minstarapproxi8Deg1Clip";
    DecoderImplementation::Minstarapproxi8JonesDeg1Clip, Minstarapproxi8JonesDeg1Clip, flooding, "Minstarapproxi8JonesDeg1Clip";
    DecoderImplementation::Minstarapproxi8PartialHardLimitDeg1Clip, Minstarapproxi8PartialHardLimitDeg1Clip, flooding, "Minstarapproxi8PartialHardLimitDeg1Clip";
    DecoderImplementation::Minstarapproxi8JonesPartialHardLimitDeg1Clip, Minstarapproxi8JonesPartialHardLimitDeg1Clip, flooding, "Minstarapproxi8JonesPartialHardLimitDeg1Clip";
    DecoderImplementation::Aminstarf64, Aminstarf64, flooding, "Aminstarf64";
    DecoderImplementation::Aminstarf32, Aminstarf32, flooding, "Aminstarf32";
    DecoderImplementation::Aminstari8, Aminstari8, flooding, "Aminstari8";
    DecoderImplementation::Aminstari8Jones, Aminstari8Jones, flooding, "Aminstari8Jones";
    DecoderImplementation::Aminstari8PartialHardLimit, Aminstari8PartialHardLimit, flooding, "Aminstari8PartialHardLimit";
    DecoderImplementation::Aminstari8JonesPartialHardLimit, Aminstari8JonesPartialHardLimit, flooding, "Aminstari8JonesPartialHardLimit";
    DecoderImplementation::Aminstari8Deg1Clip, Aminstari8Deg1Clip, flooding, "Aminstari8Deg1Clip";
    DecoderImplementation::Aminstari8JonesDeg1Clip, Aminstari8JonesDeg1Clip, flooding, "Aminstari8JonesDeg1Clip";
    DecoderImplementation::Aminstari8PartialHardLimitDeg1Clip, Aminstari8PartialHardLimitDeg1Clip, flooding, "Aminstari8PartialHardLimitDeg1Clip";
    DecoderImplementation::Aminstari8JonesPartialHardLimitDeg1Clip, Aminstari8JonesPartialHardLimitDeg1Clip, flooding, "Aminstari8JonesPartialHardLimitDeg1Clip";
    DecoderImplementation::HLPhif64, Phif64, horizontal_layered, "HLPhif64";
    DecoderImplementation::HLPhif32, Phif32, horizontal_layered, "HLPhif32";
    DecoderImplementation::HLTanhf64, Tanhf64, horizontal_layered, "HLTanhf64";
    DecoderImplementation::HLTanhf32, Tanhf32, horizontal_layered, "HLTanhf32";
    DecoderImplementation::HLMinstarapproxf64, Minstarapproxf64, horizontal_layered, "HLMinstarapproxf64";
    DecoderImplementation::HLMinstarapproxf32, Minstarapproxf32, horizontal_layered, "HLMinstarapproxf32";
    DecoderImplementation::HLMinstarapproxi8, Minstarapproxi8, horizontal_layered, "HLMinstarapproxi8";
    DecoderImplementation::HLMinstarapproxi8PartialHardLimit, Minstarapproxi8PartialHardLimit, horizontal_layered, "HLMinstarapproxi8PartialHardLimit";
    DecoderImplementation::HLAminstarf64, Aminstarf64, horizontal_layered, "HLAminstarf64";
    DecoderImplementation::HLAminstarf32, Aminstarf32, horizontal_layered, "HLAminstarf32";
    DecoderImplementation::HLAminstari8, Aminstari8, horizontal_layered, "HLAminstari8";
    DecoderImplementation::HLAminstari8PartialHardLimit, Aminstari8, horizontal_layered, "HLAminstari8PartialHardLimit";
);
