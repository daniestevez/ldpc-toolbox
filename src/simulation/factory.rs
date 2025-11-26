//! BER test factory.
//!
//! This module contains a factory that generates BER test objects as a boxed
//! trait object using the [`BerTestBuilder`].

use super::{
    ber::{BerTest, BerTestParameters, Statistics},
    modulation::{Bpsk, Psk8},
};
use crate::decoder::factory::{DecoderFactory, DecoderImplementation};
use clap::ValueEnum;

/// BER test.
///
/// This trait is used to define trait objects that implement BER tests.
pub trait Ber {
    /// Runs the BER test.
    ///
    /// This function runs the BER test until completion. It returns a list of
    /// statistics for each Eb/N0, or an error.
    fn run(self: Box<Self>) -> Result<Vec<Statistics>, Box<dyn std::error::Error>>;

    /// Returns the frame size of the code.
    ///
    /// This corresponds to the frame size after puncturing.
    fn n(&self) -> usize;

    /// Returns the codeword size of the code.
    ///
    /// This corresponds to the codeword size before puncturing.
    fn n_cw(&self) -> usize;

    /// Returns the number of information bits of the code.
    fn k(&self) -> usize;

    /// Returns the rate of the code.
    fn rate(&self) -> f64;
}

/// BER test builder.
///
/// This struct contains all the parameters needed to create a BER test.
#[derive(Debug)]
pub struct BerTestBuilder<'a, Dec = DecoderImplementation> {
    /// BER test parameters.
    pub parameters: BerTestParameters<'a, Dec>,
    /// Modulation.
    pub modulation: Modulation,
}

/// Modulation.
///
/// This enum represents the modulations that can be simulated.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, ValueEnum)]
#[clap(rename_all = "UPPER")]
pub enum Modulation {
    /// BPSK modulation.
    Bpsk,
    /// 8PSK modulation.
    Psk8,
}

impl std::str::FromStr for Modulation {
    type Err = String;

    fn from_str(s: &str) -> Result<Modulation, String> {
        Ok(match s {
            "BPSK" => Modulation::Bpsk,
            "8PSK" => Modulation::Psk8,
            _ => Err(format!("invalid modulation {s}"))?,
        })
    }
}

impl std::fmt::Display for Modulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{}",
            match self {
                Modulation::Bpsk => "BPSK",
                Modulation::Psk8 => "8PSK",
            }
        )
    }
}

impl<Dec: DecoderFactory> BerTestBuilder<'_, Dec> {
    /// Create a BER test.
    ///
    /// This function only defines the BER test. To run it it is necessary to
    /// call the [`Ber::run`] method.
    pub fn build(self) -> Result<Box<dyn Ber>, Box<dyn std::error::Error>> {
        macro_rules! impl_match {
            ($($modulation:ident),*) => {
                match self.modulation {
                    $(
                        Modulation::$modulation => Box::new(BerTest::<$modulation, Dec>::new(
                            self.parameters
                        )?),
                    )*
                }
            }
        }

        Ok(impl_match!(Bpsk, Psk8))
    }
}
