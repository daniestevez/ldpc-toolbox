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
use std::{
    sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
    time::{Duration, Instant},
};

/// BER test.
///
/// This struct is used to configure and run a BER test.
#[derive(Debug)]
pub struct BerTest {
    num_workers: usize,
    k: usize,
    rate: f64,
    encoder: Encoder,
    puncturer: Option<Puncturer>,
    modulator: BpskModulator,
    decoder: Decoder,
    ebn0s_db: Vec<f32>,
    statistics: Vec<Statistics>,
    max_iterations: usize,
    max_frame_errors: u64,
}

#[derive(Debug)]
struct Worker {
    terminate_rx: Receiver<()>,
    results_tx: Sender<WorkerResult>,
    k: usize,
    encoder: Encoder,
    puncturer: Option<Puncturer>,
    modulator: BpskModulator,
    channel: AwgnChannel,
    demodulator: BpskDemodulator,
    decoder: Decoder,
    max_iterations: usize,
}

#[derive(Debug, Clone)]
struct WorkerResultOk {
    bit_errors: u64,
    frame_error: bool,
    false_decode: bool,
}

type WorkerResult = Result<WorkerResultOk, ()>;

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
        let k = h.num_cols() - h.num_rows();
        let n = h.num_cols();
        let puncturer = puncturing_pattern.map(Puncturer::new);
        let puncturer_rate = if let Some(p) = puncturer.as_ref() {
            p.rate()
        } else {
            1.0
        };
        let rate = puncturer_rate * k as f64 / n as f64;
        Ok(BerTest {
            num_workers: num_cpus::get(),
            k,
            rate,
            encoder: Encoder::from_h(&h)?,
            puncturer,
            modulator: BpskModulator::new(),
            decoder: Decoder::new(h),
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
    pub fn run(mut self) -> Result<Vec<Statistics>, Box<dyn std::error::Error>> {
        for &ebn0_db in &self.ebn0s_db {
            let ebn0 = 10.0_f64.powf(0.1 * f64::from(ebn0_db));
            let esn0 = self.rate * ebn0;
            let noise_sigma = (0.5 / esn0).sqrt() as f32;
            let (results_tx, results_rx) = mpsc::channel();
            let workers = std::iter::repeat_with(|| {
                let (mut worker, terminate_tx) = self.make_worker(noise_sigma, results_tx.clone());
                let handle = std::thread::spawn(move || worker.work());
                (handle, terminate_tx)
            })
            .take(self.num_workers)
            .collect::<Vec<_>>();

            let mut current_statistics = CurrentStatistics::new();
            while current_statistics.frame_errors < self.max_frame_errors {
                match results_rx.recv().unwrap() {
                    Ok(result) => {
                        current_statistics.bit_errors += result.bit_errors;
                        current_statistics.frame_errors += u64::from(result.frame_error);
                        current_statistics.false_decodes += u64::from(result.false_decode);
                        current_statistics.num_frames += 1;
                    }
                    Err(()) => break,
                }
            }

            for (_, terminate_tx) in workers.iter() {
                // we don't care if this fails because the worker has terminated
                // and dropped the channel.
                let _ = terminate_tx.send(());
            }

            for (handle, _) in workers.into_iter() {
                // This cannot be written with the question mark because the
                // error isn't Sized.
                #[allow(clippy::question_mark)]
                if let Err(e) = handle.join().unwrap() {
                    return Err(e);
                }
            }

            self.statistics.push(Statistics::from_current(
                &current_statistics,
                ebn0_db,
                self.k,
            ));
        }
        Ok(self.statistics)
    }

    fn make_worker(
        &self,
        noise_sigma: f32,
        results_tx: Sender<WorkerResult>,
    ) -> (Worker, SyncSender<()>) {
        let (terminate_tx, terminate_rx) = mpsc::sync_channel(1);
        (
            Worker {
                terminate_rx,
                results_tx,
                k: self.k,
                encoder: self.encoder.clone(),
                puncturer: self.puncturer.clone(),
                modulator: self.modulator.clone(),
                channel: AwgnChannel::new(noise_sigma),
                demodulator: BpskDemodulator::new(noise_sigma),
                decoder: self.decoder.clone(),
                max_iterations: self.max_iterations,
            },
            terminate_tx,
        )
    }
}

impl Worker {
    fn work(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut rng = rand::thread_rng();
        loop {
            match self.terminate_rx.try_recv() {
                Ok(()) => return Ok(()),
                Err(TryRecvError::Disconnected) => panic!(),
                Err(TryRecvError::Empty) => (),
            };
            let result = self.simulate(&mut rng);
            let to_send = match result.as_ref() {
                Ok(r) => Ok(r.clone()),
                Err(_) => Err(()),
            };
            self.results_tx.send(to_send).unwrap();
            result?;
        }
    }

    fn simulate<R: Rng>(
        &mut self,
        rng: &mut R,
    ) -> Result<WorkerResultOk, Box<dyn std::error::Error + Send + Sync + 'static>> {
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

        let (decoded, success) = match self.decoder.decode(&llrs_decoder, self.max_iterations) {
            Ok((d, _)) => (d, true),
            Err(d) => (d, false),
        };
        // Count only bit errors in the systematic part of the codeword
        let bit_errors = message
            .iter()
            .zip(decoded.iter())
            .filter(|(&a, &b)| a != b)
            .count() as u64;
        let frame_error = bit_errors > 0;
        let false_decode = frame_error && success;
        Ok(WorkerResultOk {
            bit_errors,
            frame_error,
            false_decode,
        })
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
