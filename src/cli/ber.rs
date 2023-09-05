//! BER test CLI subcommand.
//!
//! This subcommand can be used to perform a BER test of an LDPC decoder.
//!
//! # Examples
//!
//! The CCSDS r=1/2, k=1024 LDPC code can be simulated with
//! ```shell
//! $ ldpc-toolbox ber --min-ebn0 0.0 --max-ebn0 2.05 --step-ebn0 0.1 \
//!       --puncturing 1,1,1,1,0 ar4ja_1_2_1024.alist
//! ```
//!
//! The alist file must have been generated previoulsy with the
//! [ccsds](super::ccsds) subcommand.

use crate::{
    cli::*,
    decoder::factory::DecoderImplementation,
    simulation::ber::{BerTest, Report, Reporter, Statistics},
    sparse::SparseMatrix,
};
use clap::Parser;
use console::Term;
use std::{
    error::Error,
    fs::File,
    io::Write,
    sync::mpsc::{self, Receiver},
    time::Duration,
};

/// BER test CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Performs a BER simulation")]
pub struct Args {
    /// alist file for the code
    alist: String,
    /// Output file for simulation results
    #[structopt(long)]
    output_file: Option<String>,
    /// Decoder implementation
    #[structopt(long, default_value = "Phif64")]
    decoder: DecoderImplementation,
    /// Puncturing pattern (format "1,1,1,0")
    #[structopt(long)]
    puncturing: Option<String>,
    /// Minimum Eb/N0 (dB)
    #[structopt(long)]
    min_ebn0: f64,
    /// Maximum Eb/N0 (dB)
    #[structopt(long)]
    max_ebn0: f64,
    /// Eb/N0 step (dB)
    #[structopt(long)]
    step_ebn0: f64,
    /// Maximum number of iterations
    #[structopt(long, default_value = "100")]
    max_iter: usize,
    /// Number of frame errors to collect
    #[structopt(long, default_value = "100")]
    frame_errors: u64,
}

impl Run for Args {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let puncturing_pattern = if let Some(p) = self.puncturing.as_ref() {
            Some(parse_puncturing_pattern(p)?)
        } else {
            None
        };
        let h = SparseMatrix::from_alist(&std::fs::read_to_string(&self.alist)?)?;
        let output_file = if let Some(f) = &self.output_file {
            Some(File::create(f)?)
        } else {
            None
        };
        let num_ebn0s = ((self.max_ebn0 - self.min_ebn0) / self.step_ebn0).floor() as usize + 1;
        let ebn0s = (0..num_ebn0s)
            .map(|k| (self.min_ebn0 + k as f64 * self.step_ebn0) as f32)
            .collect::<Vec<_>>();
        let (report_tx, report_rx) = mpsc::channel();
        let reporter = Reporter {
            tx: report_tx,
            interval: Duration::from_millis(500),
        };
        let test = BerTest::new(
            h,
            self.decoder,
            puncturing_pattern.as_ref().map(|v| &v[..]),
            self.frame_errors,
            self.max_iter,
            &ebn0s,
            Some(reporter),
        )?;
        self.write_details(std::io::stdout(), &test)?;
        if let Some(f) = &output_file {
            self.write_details(f, &test)?;
        }
        let mut progress = Progress::new(report_rx, output_file);
        let progress = std::thread::spawn(move || progress.run());
        test.run()?;
        // This block cannot actually be written with the ? operator
        #[allow(clippy::question_mark)]
        if let Err(e) = progress.join().unwrap() {
            return Err(e);
        }
        Ok(())
    }
}

impl Args {
    fn write_details<W: Write>(&self, mut f: W, test: &BerTest) -> std::io::Result<()> {
        writeln!(f, "BER TEST PARAMETERS")?;
        writeln!(f, "-------------------")?;
        writeln!(f, "Simulation:")?;
        writeln!(f, " - Minimum Eb/N0: {:.2} dB", self.min_ebn0)?;
        writeln!(f, " - Maximum Eb/N0: {:.2} dB", self.max_ebn0)?;
        writeln!(f, " - Eb/N0 step: {:.2} dB", self.step_ebn0)?;
        writeln!(f, " - Number of frame errors: {}", self.frame_errors)?;
        writeln!(f, "LDPC code:")?;
        writeln!(f, " - alist: {}", self.alist)?;
        if let Some(puncturing) = self.puncturing.as_ref() {
            writeln!(f, " - Puncturing pattern: {puncturing}")?;
        }
        writeln!(f, " - Information bits (k): {}", test.k())?;
        writeln!(f, " - Codeword size (N_cw): {}", test.n_cw())?;
        writeln!(f, " - Frame size (N): {}", test.n())?;
        writeln!(f, " - Code rate: {:.3}", test.rate())?;
        writeln!(f, "LDPC decoder:")?;
        writeln!(f, " - Implementation: {}", self.decoder)?;
        writeln!(f, " - Maximum iterations: {}", self.max_iter)?;
        writeln!(f)?;
        Ok(())
    }
}

/// Parses a puncturing pattern.
///
/// This function parses a punturing pattern given as a string, converting it
/// into a vector of bools. The format for the puncturing pattern should be
/// like `"1,1,1,0"`.
pub fn parse_puncturing_pattern(s: &str) -> Result<Vec<bool>, &'static str> {
    let mut v = Vec::new();
    for a in s.split(',') {
        v.push(match a {
            "0" => false,
            "1" => true,
            _ => return Err("invalid puncturing pattern"),
        });
    }
    Ok(v)
}

#[derive(Debug)]
struct Progress {
    rx: Receiver<Report>,
    term: Term,
    output_file: Option<File>,
}

impl Progress {
    fn new(rx: Receiver<Report>, output_file: Option<File>) -> Progress {
        Progress {
            rx,
            term: Term::stdout(),
            output_file,
        }
    }

    fn run(&mut self) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
        ctrlc::set_handler({
            let term = self.term.clone();
            move || {
                let _ = term.write_line("");
                let _ = term.show_cursor();
                std::process::exit(0);
            }
        })?;

        let ret = self.work();
        self.term.write_line("")?;
        self.term.show_cursor()?;
        ret
    }

    fn work(&mut self) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
        self.term.set_title("ldpc-toolbox ber");
        self.term.hide_cursor()?;
        self.term.write_line(Self::format_header())?;
        if let Some(f) = &mut self.output_file {
            writeln!(f, "{}", Self::format_header())?;
        }
        let mut last_stats = None;
        loop {
            let Report::Statistics(stats) = self.rx.recv().unwrap() else {
                // BER test has finished
                if let Some(f) = &mut self.output_file {
                    writeln!(f, "{}", &Self::format_progress(&last_stats.unwrap()))?;
                }
                return Ok(());
            };
            if let Some(s) = &last_stats {
                if s.ebn0_db != stats.ebn0_db {
                    if let Some(f) = &mut self.output_file {
                        writeln!(f, "{}", &Self::format_progress(s))?;
                    }
                }
            }
            match &last_stats {
                Some(s) if s.ebn0_db == stats.ebn0_db => {
                    self.term.move_cursor_up(1)?;
                    self.term.clear_line()?;
                }
                _ => (),
            };
            self.term.write_line(&Self::format_progress(&stats))?;
            last_stats = Some(stats);
        }
    }

    fn format_header() -> &'static str {
        "  Eb/N0 |   Frames | Bit errs | Frame er | False de |     BER |     FER | Avg iter | Avg corr | Throughp | Elapsed\n\
         --------|----------|----------|----------|----------|---------|---------|----------|----------|----------|----------"
    }

    fn format_progress(stats: &Statistics) -> String {
        format!(
            "{:7.2} | {:8} | {:8} | {:8} | {:8} | {:7.2e} | {:7.2e} | {:8.1} | {:8.1} | {:8.3} | {}",
            stats.ebn0_db,
            stats.num_frames,
            stats.bit_errors,
            stats.frame_errors,
            stats.false_decodes,
            stats.ber,
            stats.fer,
            stats.average_iterations,
            stats.average_iterations_correct,
            stats.throughput_mbps,
            humantime::format_duration(Duration::from_secs(stats.elapsed.as_secs()))
        )
    }
}
