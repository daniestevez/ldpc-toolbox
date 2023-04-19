//! BER simulation
//!
//! This module contains utilities for BER simulation.

use super::{
    channel::AwgnChannel,
    modulation::{BpskDemodulator, BpskModulator},
    puncturing::Puncturer,
};
use crate::{
    decoder::factory::{DecoderImplementation, LdpcDecoder},
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
    decoder_implementation: DecoderImplementation,
    h: SparseMatrix,
    num_workers: usize,
    k: usize,
    n: usize,
    n_cw: usize,
    rate: f64,
    encoder: Encoder,
    puncturer: Option<Puncturer>,
    modulator: BpskModulator,
    ebn0s_db: Vec<f32>,
    statistics: Vec<Statistics>,
    max_iterations: usize,
    max_frame_errors: u64,
    reporter: Option<Reporter>,
    last_reported: Instant,
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
    decoder: Box<dyn LdpcDecoder>,
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

/// Progress reporter.
///
/// A reporter can optionally be supplied to the BER test on contruction in
/// order to receive periodic messages reporting the test progress.
#[derive(Debug, Clone)]
pub struct Reporter {
    /// Sender element of a channel used to send the reports.
    pub tx: Sender<Report>,
    /// Reporting interval.
    pub interval: Duration,
}

/// BER test progress report.
///
/// Progress reports are optionally sent out periodically by the BER test. These
/// can be used to update a UI to show the progress.
#[derive(Debug, Clone, PartialEq)]
pub enum Report {
    /// Statistics for the current Eb/N0 being tested.
    ///
    /// This is sent periodically, and also when the Eb/N0 is finished.
    Statistics(Statistics),
    /// The complete BER test has finished.
    ///
    /// This is sent when all the Eb/N0 cases have been done.
    Finished,
}

macro_rules! report {
    ($self:expr, $current_statistics:expr, $ebn0_db:expr, $final:expr) => {
        if let Some(reporter) = $self.reporter.as_ref() {
            let now = Instant::now();
            if $final || $self.last_reported + reporter.interval < now {
                reporter
                    .tx
                    .send(Report::Statistics(Statistics::from_current(
                        &$current_statistics,
                        $ebn0_db,
                        $self.k,
                    )))
                    .unwrap();
                $self.last_reported = now;
            }
        }
    };
}

impl BerTest {
    /// Creates a new BER test.
    ///
    /// The parameters required to define the test are the parity check matrix
    /// `h`, an optional puncturing pattern (which uses the semantics of
    /// [`Puncturer`]), the maximum number of frame errors at which to stop the
    /// simulation for each Eb/N0, the maximum number of iterations of the LDPC
    /// decoder, a list of Eb/N0's in dB units, and an optional [`Reporter`] to
    /// send messages about the test progress.
    ///
    /// This function only defines the BER test. To run it it is necessary to
    /// call the [`BerTest::run`] method.
    pub fn new(
        h: SparseMatrix,
        decoder_implementation: DecoderImplementation,
        puncturing_pattern: Option<&[bool]>,
        max_frame_errors: u64,
        max_iterations: usize,
        ebn0s_db: &[f32],
        reporter: Option<Reporter>,
    ) -> Result<BerTest, Error> {
        let k = h.num_cols() - h.num_rows();
        let n_cw = h.num_cols();
        let puncturer = puncturing_pattern.map(Puncturer::new);
        let puncturer_rate = if let Some(p) = puncturer.as_ref() {
            p.rate()
        } else {
            1.0
        };
        let n = (n_cw as f64 / puncturer_rate).round() as usize;
        let rate = k as f64 / n as f64;
        Ok(BerTest {
            decoder_implementation,
            num_workers: num_cpus::get(),
            k,
            n,
            n_cw,
            rate,
            encoder: Encoder::from_h(&h)?,
            h,
            puncturer,
            modulator: BpskModulator::new(),
            ebn0s_db: ebn0s_db.to_owned(),
            statistics: Vec::with_capacity(ebn0s_db.len()),
            max_iterations,
            max_frame_errors,
            reporter,
            last_reported: Instant::now(),
        })
    }

    /// Runs the BER test.
    ///
    /// This function runs the BER test until completion. It returns a list of
    /// statistics for each Eb/N0, or an error.
    pub fn run(mut self) -> Result<Vec<Statistics>, Box<dyn std::error::Error>> {
        let ret = self.do_run();
        if let Some(reporter) = self.reporter.as_ref() {
            reporter.tx.send(Report::Finished).unwrap();
        }
        ret?;
        Ok(self.statistics)
    }

    /// Returns the frame size of the code.
    ///
    /// This corresponds to the frame size after puncturing.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the codeword size of the code.
    ///
    /// This corresponds to the codeword size before puncturing.
    pub fn n_cw(&self) -> usize {
        self.n_cw
    }

    /// Returns the number of information bits of the code.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the rate of the code.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    fn do_run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.last_reported = Instant::now();
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
                report!(self, current_statistics, ebn0_db, false);
            }
            report!(self, current_statistics, ebn0_db, true);

            for (_, terminate_tx) in workers.iter() {
                // we don't care if this fails because the worker has terminated
                // and dropped the channel.
                let _ = terminate_tx.send(());
            }

            let mut join_error = None;
            for (handle, _) in workers.into_iter() {
                if let Err(e) = handle.join().unwrap() {
                    join_error = Some(e);
                }
            }
            if let Some(e) = join_error {
                return Err(e);
            }

            self.statistics.push(Statistics::from_current(
                &current_statistics,
                ebn0_db,
                self.k,
            ));
        }
        Ok(())
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
                decoder: self.decoder_implementation.build_decoder(self.h.clone()),
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
            Ok(output) => (output.codeword, true),
            Err(output) => (output.codeword, false),
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
