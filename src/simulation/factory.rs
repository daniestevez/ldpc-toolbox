//! BER test factory.
//!
//! This module contains a factory that generates BER test objects as a boxed
//! trait object using the [`BerTestBuilder`].

use super::{
    ber::{BerTest, Reporter, Statistics},
    modulation::{Bpsk, Psk8},
};
use crate::{decoder::factory::DecoderImplementation, sparse::SparseMatrix};

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
pub struct BerTestBuilder<'a> {
    /// LDPC parity check matrix.
    pub h: SparseMatrix,
    /// LDPC decoder implementation.
    pub decoder_implementation: DecoderImplementation,
    /// Modulation.
    pub modulation: Modulation,
    /// Codeword puncturing pattern.
    pub puncturing_pattern: Option<&'a [bool]>,
    /// Maximum number of frame errors per Eb/N0.
    pub max_frame_errors: u64,
    /// Maximum number of iterations per codeword.
    pub max_iterations: usize,
    /// List of Eb/N0's (in dB) to simulate.
    pub ebn0s_db: &'a [f32],
    /// An optional reporter object to which the BER test will send periodic
    /// updates about its progress.
    pub reporter: Option<Reporter>,
}

/// Modulation.
///
/// This enum represents the modulations that can be simulated.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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

impl<'a> BerTestBuilder<'a> {
    /// Create a BER test.
    ///
    /// This function only defines the BER test. To run it it is necessary to
    /// call the [`Ber::run`] method.
    pub fn build(self) -> Result<Box<dyn Ber>, Box<dyn std::error::Error>> {
        Ok(match self.modulation {
            Modulation::Bpsk => Box::new(BerTest::<Bpsk>::new(
                self.h,
                self.decoder_implementation,
                self.puncturing_pattern,
                self.max_frame_errors,
                self.max_iterations,
                self.ebn0s_db,
                self.reporter,
            )?),
            Modulation::Psk8 => Box::new(BerTest::<Psk8>::new(
                self.h,
                self.decoder_implementation,
                self.puncturing_pattern,
                self.max_frame_errors,
                self.max_iterations,
                self.ebn0s_db,
                self.reporter,
            )?),
        })
    }
}