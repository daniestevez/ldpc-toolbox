//! BER simulation
//!
//! This module contains utilities for BER simulation.

use super::{
    channel::{AwgnChannel, Channel},
    factory::Ber,
    interleaving::Interleaver,
    modulation::{Demodulator, Modulation, Modulator},
    puncturing::Puncturer,
};
use crate::{
    decoder::{
        LdpcDecoder,
        factory::{DecoderFactory, DecoderImplementation},
    },
    encoder::{Encoder, Error},
    gf2::GF2,
    sparse::SparseMatrix,
};
use ndarray::Array1;
use num_traits::{One, Zero};
use rand::{Rng, distributions::Standard};
use std::{
    sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
    time::{Duration, Instant},
};

/// BER test.
///
/// This struct is used to configure and run a BER test.
#[derive(Debug)]
pub struct BerTest<Mod: Modulation, Dec = DecoderImplementation> {
    decoder_implementation: Dec,
    h: SparseMatrix,
    num_workers: usize,
    k: usize,
    n: usize,
    n_cw: usize,
    rate: f64,
    encoder: Encoder,
    puncturer: Option<Puncturer>,
    interleaver: Option<Interleaver>,
    modulator: Mod::Modulator,
    ebn0s_db: Vec<f32>,
    statistics: Vec<Statistics>,
    bch_max_errors: u64,
    max_iterations: usize,
    max_frame_errors: u64,
    reporter: Option<Reporter>,
    last_reported: Instant,
}

#[derive(Debug)]
struct Worker<Mod: Modulation> {
    terminate_rx: Receiver<()>,
    results_tx: Sender<WorkerResult>,
    k: usize,
    encoder: Encoder,
    puncturer: Option<Puncturer>,
    interleaver: Option<Interleaver>,
    modulator: Mod::Modulator,
    channel: AwgnChannel,
    demodulator: Mod::Demodulator,
    decoder: Box<dyn LdpcDecoder>,
    max_iterations: usize,
}

#[derive(Debug, Clone)]
struct WorkerResultOk {
    bit_errors: u64,
    frame_error: bool,
    false_decode: bool,
    iterations: u64,
}

type WorkerResult = Result<WorkerResultOk, ()>;

#[derive(Debug, Clone, PartialEq)]
struct CurrentStatistics {
    num_frames: u64,
    false_decodes: u64,
    total_iterations: u64,
    start: Instant,
    ldpc: CurrentCodeStatistics,
    bch: Option<CurrentCodeStatistics>,
}

#[derive(Debug, Clone, PartialEq)]
struct CurrentCodeStatistics {
    bit_errors: u64,
    frame_errors: u64,
    correct_iterations: u64,
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
    /// Total number of iterations.
    pub total_iterations: u64,
    /// Number of frames falsely decoded.
    ///
    /// These are frames for which the decoder has converged to a valid
    /// codeword, but the codeword is different from the transmitted codeword.
    pub false_decodes: u64,
    /// Average iterations per frame.
    pub average_iterations: f64,
    /// Elapsed time for this test case.
    pub elapsed: Duration,
    /// Throughput in Mbps (referred to information bits).
    pub throughput_mbps: f64,
    /// Statistics of the inner LDPC decoder.
    pub ldpc: CodeStatistics,
    /// Statistics of the combined inner LDPC decoder plus outer BCH decoder (if it exists).
    pub bch: Option<CodeStatistics>,
}

/// BER test statistics for a particular code.
///
/// This is a substructure of [`Statistics`] that includes all the elements that
/// depend on whether we are using only the inner LDPC decoder or LDPC plus
/// BCH. `Statistics` has two instances of this structure: one for LDPC-only and
/// another for LDPC plus BCH (which is only present when BCH is enabled).
#[derive(Debug, Clone, PartialEq)]
pub struct CodeStatistics {
    /// Number of bit errors.
    pub bit_errors: u64,
    /// Number of frame errors.
    pub frame_errors: u64,
    /// Sum of iterations in correct frames.
    pub correct_iterations: u64,
    /// Bit error rate.
    pub ber: f64,
    /// Frame error rate.
    pub fer: f64,
    /// Average iterations per correct frame.
    pub average_iterations_correct: f64,
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

impl<Mod: Modulation, Dec: DecoderFactory> BerTest<Mod, Dec> {
    /// Creates a new BER test.
    ///
    /// The parameters required to define the test are the parity check matrix
    /// `h`, an optional puncturing pattern (which uses the semantics of
    /// [`Puncturer`]), an optional interleaving pattern, the maximum number of
    /// frame errors at which to stop the simulation for each Eb/N0, the maximum
    /// number of iterations of the LDPC decoder, a list of Eb/N0's in dB units,
    /// and an optional [`Reporter`] to send messages about the test progress.
    ///
    /// This function only defines the BER test. To run it it is necessary to
    /// call the [`BerTest::run`] method.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        h: SparseMatrix,
        decoder_implementation: Dec,
        puncturing_pattern: Option<&[bool]>,
        interleaving_columns: Option<isize>,
        max_frame_errors: u64,
        max_iterations: usize,
        ebn0s_db: &[f32],
        reporter: Option<Reporter>,
        bch_max_errors: u64,
    ) -> Result<BerTest<Mod, Dec>, Error> {
        let k = h.num_cols() - h.num_rows();
        let n_cw = h.num_cols();
        let puncturer = puncturing_pattern.map(Puncturer::new);
        let interleaver = interleaving_columns.map(|n| Interleaver::new(n.unsigned_abs(), n < 0));
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
            interleaver,
            modulator: Mod::Modulator::default(),
            ebn0s_db: ebn0s_db.to_owned(),
            statistics: Vec::with_capacity(ebn0s_db.len()),
            bch_max_errors,
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

    fn do_run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.last_reported = Instant::now();
        for &ebn0_db in &self.ebn0s_db {
            let ebn0 = 10.0_f64.powf(0.1 * f64::from(ebn0_db));
            let esn0 = self.rate * Mod::BITS_PER_SYMBOL * ebn0;
            let noise_sigma = (0.5 / esn0).sqrt();
            let (results_tx, results_rx) = mpsc::channel();
            let workers = std::iter::repeat_with(|| {
                let (mut worker, terminate_tx) = self.make_worker(noise_sigma, results_tx.clone());
                let handle = std::thread::spawn(move || worker.work());
                (handle, terminate_tx)
            })
            .take(self.num_workers)
            .collect::<Vec<_>>();

            let mut current_statistics = CurrentStatistics::new(self.bch_max_errors > 0);
            while current_statistics.errors_for_termination() < self.max_frame_errors {
                match results_rx.recv().unwrap() {
                    Ok(result) => {
                        current_statistics.ldpc.bit_errors += result.bit_errors;
                        current_statistics.ldpc.frame_errors += u64::from(result.frame_error);
                        current_statistics.false_decodes += u64::from(result.false_decode);
                        current_statistics.total_iterations += result.iterations;
                        if !result.frame_error {
                            current_statistics.ldpc.correct_iterations += result.iterations;
                        }
                        current_statistics.num_frames += 1;
                        if let Some(bch) = &mut current_statistics.bch {
                            if result.bit_errors > self.bch_max_errors {
                                // BCH cannot decode codeword
                                bch.bit_errors += result.bit_errors;
                                bch.frame_errors += 1;
                            } else {
                                // BCH can decode codeword
                                bch.correct_iterations += result.iterations;
                            }
                        }
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
        noise_sigma: f64,
        results_tx: Sender<WorkerResult>,
    ) -> (Worker<Mod>, SyncSender<()>) {
        let (terminate_tx, terminate_rx) = mpsc::sync_channel(1);
        (
            Worker {
                terminate_rx,
                results_tx,
                k: self.k,
                encoder: self.encoder.clone(),
                puncturer: self.puncturer.clone(),
                interleaver: self.interleaver.clone(),
                modulator: self.modulator.clone(),
                channel: AwgnChannel::new(noise_sigma),
                demodulator: Mod::Demodulator::from_noise_sigma(noise_sigma),
                decoder: self.decoder_implementation.build_decoder(self.h.clone()),
                max_iterations: self.max_iterations,
            },
            terminate_tx,
        )
    }
}

impl<Mod: Modulation, Dec: DecoderFactory> Ber for BerTest<Mod, Dec> {
    fn run(self: Box<Self>) -> Result<Vec<Statistics>, Box<dyn std::error::Error>> {
        BerTest::run(*self)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn n_cw(&self) -> usize {
        self.n_cw
    }

    fn k(&self) -> usize {
        self.k
    }

    fn rate(&self) -> f64 {
        self.rate
    }
}

impl<Mod: Modulation> Worker<Mod> {
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
        let transmitted = match self.interleaver.as_ref() {
            Some(i) => i.interleave(&transmitted),
            None => transmitted,
        };
        let mut symbols = self.modulator.modulate(&transmitted);
        self.channel.add_noise(rng, &mut symbols);
        let llrs_demod = self.demodulator.demodulate(&symbols);
        let llrs_decoder = match self.interleaver.as_ref() {
            Some(i) => i.deinterleave(&llrs_demod),
            None => llrs_demod,
        };
        let llrs_decoder = match self.puncturer.as_ref() {
            Some(p) => p.depuncture(&llrs_decoder)?,
            None => llrs_decoder,
        };

        let (decoded, iterations, success) =
            match self.decoder.decode(&llrs_decoder, self.max_iterations) {
                Ok(output) => (output.codeword, output.iterations, true),
                Err(output) => (output.codeword, output.iterations, false),
            };
        // Count only bit errors in the systematic part of the codeword
        let bit_errors = message
            .iter()
            .zip(decoded.iter())
            .filter(|(a, b)| a != b)
            .count() as u64;
        let frame_error = bit_errors > 0;
        let false_decode = frame_error && success;
        Ok(WorkerResultOk {
            bit_errors,
            frame_error,
            false_decode,
            iterations: iterations as u64,
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
    fn new(has_bch: bool) -> CurrentStatistics {
        CurrentStatistics {
            num_frames: 0,
            false_decodes: 0,
            total_iterations: 0,
            start: Instant::now(),
            ldpc: CurrentCodeStatistics::new(),
            bch: if has_bch {
                Some(CurrentCodeStatistics::new())
            } else {
                None
            },
        }
    }

    fn errors_for_termination(&self) -> u64 {
        if let Some(bch) = &self.bch {
            bch.frame_errors
        } else {
            self.ldpc.frame_errors
        }
    }
}

impl CurrentCodeStatistics {
    fn new() -> CurrentCodeStatistics {
        CurrentCodeStatistics {
            bit_errors: 0,
            frame_errors: 0,
            correct_iterations: 0,
        }
    }
}

impl Default for CurrentCodeStatistics {
    fn default() -> CurrentCodeStatistics {
        CurrentCodeStatistics::new()
    }
}

impl CodeStatistics {
    fn from_current(stats: &CurrentCodeStatistics, num_frames: u64, k: usize) -> CodeStatistics {
        CodeStatistics {
            bit_errors: stats.bit_errors,
            frame_errors: stats.frame_errors,
            correct_iterations: stats.correct_iterations,
            ber: stats.bit_errors as f64 / (k as f64 * num_frames as f64),
            fer: stats.frame_errors as f64 / num_frames as f64,
            average_iterations_correct: stats.correct_iterations as f64
                / (num_frames - stats.frame_errors) as f64,
        }
    }
}

impl Statistics {
    fn from_current(stats: &CurrentStatistics, ebn0_db: f32, k: usize) -> Statistics {
        let elapsed = Instant::now() - stats.start;
        Statistics {
            ebn0_db,
            num_frames: stats.num_frames,
            false_decodes: stats.false_decodes,
            total_iterations: stats.total_iterations,
            average_iterations: stats.total_iterations as f64 / stats.num_frames as f64,
            elapsed,
            throughput_mbps: 1e-6 * (k as f64 * stats.num_frames as f64) / elapsed.as_secs_f64(),
            ldpc: CodeStatistics::from_current(&stats.ldpc, stats.num_frames, k),
            bch: stats
                .bch
                .as_ref()
                .map(|b| CodeStatistics::from_current(b, stats.num_frames, k)),
        }
    }
}
