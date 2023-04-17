//! BER simulation
//!
//! This module contains utilities for BER simulation.

use super::{
    channel::AwgnChannel,
    modulation::{BpskDemodulator, BpskModulator},
    puncturing::Puncturer,
};
use crate::{
    decoder::Decoder,
    encoder::{Encoder, Error},
    gf2::GF2,
    sparse::SparseMatrix,
};
use ndarray::Array1;
use num_traits::{One, Zero};
use rand::{distributions::Standard, Rng};
use std::time::{Duration, Instant};

/// BER test.
///
/// This struct is used to configure and run a BER test.
#[derive(Debug)]
pub struct BerTest {
    k: usize,
    n: usize,
    encoder: Encoder,
    puncturer: Option<Puncturer>,
    modulator: BpskModulator,
    channel: AwgnChannel,
    demodulator: BpskDemodulator,
    decoder: Decoder,
    current_statistics: CurrentStatistics,
    ebn0s_db: Vec<f32>,
    statistics: Vec<Statistics>,
    max_iterations: usize,
    max_frame_errors: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct CurrentStatistics {
    num_frames: u64,
    bit_errors: u64,
    frame_errors: u64,
    false_decodes: u64,
    start: Instant,
}

/// BER test statistics.
///
/// This structure contains the statistics for a single Eb/N0 case in a BER
/// test.
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// Eb/N0 in dB units.
    pub ebn0_db: f32,
    /// Number of frames tested.
    pub num_frames: u64,
    /// Number of bit errors.
    pub bit_errors: u64,
    /// Number of frame errors.
    pub frame_errors: u64,
    /// Number of frames falsely decoded.
    ///
    /// This are frames for which the decoder converged to a valid codeword, but
    /// the codeword is different to the transmitted codeword.
    pub false_decodes: u64,
    /// Bit error rate.
    pub ber: f64,
    /// Frame error rate.
    pub fer: f64,
    /// Elapsed time for this test case.
    pub elapsed: Duration,
    /// Throughput in Mbps (referred to information bits).
    pub throughput_mbps: f64,
}

impl BerTest {
    /// Creates a new BER test.
    ///
    /// The parameters required to define the test are the parity check matrix
    /// `h`, an optional puncturing pattern (which uses the semantics of
    /// [`Puncturer`]), the maximum number of frame errors at which to stop the
    /// simulation for each Eb/N0, the maximum number of iterations of the LDPC
    /// decoder, and a list of Eb/N0's in dB units.
    ///
    /// This function only defines the BER test. To run it it is necessary to
    /// call the [`run`] method.
    pub fn new(
        h: SparseMatrix,
        puncturing_pattern: Option<&[bool]>,
        max_frame_errors: u64,
        max_iterations: usize,
        ebn0s_db: &[f32],
    ) -> Result<BerTest, Error> {
        Ok(BerTest {
            k: h.num_cols() - h.num_rows(),
            n: h.num_cols(),
            encoder: Encoder::from_h(&h)?,
            puncturer: puncturing_pattern.map(Puncturer::new),
            modulator: BpskModulator::new(),
            // The channel and demodulator are thrown away every time we change
            // Eb/N0. We initialize to arbitrary values.
            channel: AwgnChannel::new(1.0),
            demodulator: BpskDemodulator::new(1.0),
            decoder: Decoder::new(h),
            // These statistics will be discarded when we start.
            current_statistics: CurrentStatistics::new(),
            ebn0s_db: ebn0s_db.to_owned(),
            statistics: Vec::with_capacity(ebn0s_db.len()),
            max_iterations,
            max_frame_errors,
        })
    }

    /// Runs the BER test.
    ///
    /// This function runs the BER test until completion. It returns a list of
    /// statistics for each Eb/N0, or an error.
    pub fn run<R: Rng>(
        mut self,
        rng: &mut R,
    ) -> Result<Vec<Statistics>, Box<dyn std::error::Error>> {
        let puncturer_rate = if let Some(p) = self.puncturer.as_ref() {
            p.rate()
        } else {
            1.0
        };
        let rate = puncturer_rate * self.k as f64 / self.n as f64;

        for ebn0_db in self.ebn0s_db {
            let ebn0 = 10.0_f64.powf(0.1 * f64::from(ebn0_db));
            let esn0 = rate * ebn0;
            let noise_sigma = (0.5 / esn0).sqrt() as f32;
            self.channel = AwgnChannel::new(noise_sigma);
            self.demodulator = BpskDemodulator::new(noise_sigma);
            self.current_statistics = CurrentStatistics::new();
            while self.current_statistics.frame_errors < self.max_frame_errors {
                let message = Self::random_message(rng, self.k);
                let codeword = self.encoder.encode(&Self::gf2_array(&message));
                let transmitted = match self.puncturer.as_ref() {
                    Some(p) => p.puncture(&codeword)?,
                    None => codeword,
                };
                let mut symbols = self.modulator.modulate(&transmitted);
                self.channel.add_noise(rng, &mut symbols);
                let llrs_demod = self.demodulator.demodulate(&symbols);
                let llrs_decoder = match self.puncturer.as_ref() {
                    Some(p) => p.depuncture(&llrs_demod)?,
                    None => llrs_demod,
                };

                let (decoded, success) =
                    match self.decoder.decode(&llrs_decoder, self.max_iterations) {
                        Ok((d, _)) => (d, true),
                        Err(d) => (d, false),
                    };
                // Count only bit errors in the systematic part of the codeword
                let bit_errors = message
                    .iter()
                    .zip(decoded.iter())
                    .filter(|(&a, &b)| a != b)
                    .count() as u64;
                self.current_statistics.bit_errors += bit_errors;
                if bit_errors > 0 {
                    self.current_statistics.frame_errors += 1;
                    if success {
                        self.current_statistics.false_decodes += 1;
                    }
                }
                self.current_statistics.num_frames += 1;
                dbg!(&self.current_statistics);
            }
            self.statistics.push(Statistics::from_current(
                &self.current_statistics,
                ebn0_db,
                self.k,
            ));
        }
        Ok(self.statistics)
    }

    fn random_message<R: Rng>(rng: &mut R, size: usize) -> Vec<u8> {
        rng.sample_iter(Standard)
            .map(<u8 as From<bool>>::from)
            .take(size)
            .collect()
    }

    fn gf2_array(bits: &[u8]) -> Array1<GF2> {
        Array1::from_iter(
            bits.iter()
                .map(|&b| if b == 1 { GF2::one() } else { GF2::zero() }),
        )
    }
}

impl CurrentStatistics {
    fn new() -> CurrentStatistics {
        CurrentStatistics {
            num_frames: 0,
            bit_errors: 0,
            frame_errors: 0,
            false_decodes: 0,
            start: Instant::now(),
        }
    }
}

impl Default for CurrentStatistics {
    fn default() -> CurrentStatistics {
        CurrentStatistics::new()
    }
}

impl Statistics {
    fn from_current(stats: &CurrentStatistics, ebn0_db: f32, k: usize) -> Statistics {
        let elapsed = Instant::now() - stats.start;
        Statistics {
            ebn0_db,
            num_frames: stats.num_frames,
            bit_errors: stats.bit_errors,
            frame_errors: stats.frame_errors,
            false_decodes: stats.false_decodes,
            ber: stats.bit_errors as f64 / (k as f64 * stats.num_frames as f64),
            fer: stats.frame_errors as f64 / stats.num_frames as f64,
            elapsed,
            throughput_mbps: 1e-6 * (k as f64 * stats.num_frames as f64) / elapsed.as_secs_f64(),
        }
    }
}
