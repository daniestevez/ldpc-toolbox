//! BER test CLI subcommand.
//!
//! This subcommand can be used to perform a BER test of an LDPC decoder.

use crate::{
    cli::*,
    decoder::factory::DecoderImplementation,
    simulation::ber::{BerTest, Report, Reporter, Statistics},
    sparse::SparseMatrix,
};
use clap::Parser;
use console::Term;
use std::error::Error;
use std::{
    sync::mpsc::{self, Receiver},
    time::Duration,
};

/// BER test CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Performs a BER simulation")]
pub struct Args {
    /// alist file for the code
    alist: String,
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
        self.print_details(&test);
        let progress = Progress::new(report_rx);
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
    fn print_details(&self, test: &BerTest) {
        println!("BER TEST PARAMETERS");
        println!("-------------------");
        println!("Simulation:");
        println!(" - Minimum Eb/N0: {:.2} dB", self.min_ebn0);
        println!(" - Maximum Eb/N0: {:.2} dB", self.max_ebn0);
        println!(" - Eb/N0 step: {:.2} dB", self.step_ebn0);
        println!(" - Number of frame errors: {}", self.frame_errors);
        println!("LDPC code:");
        println!(" - alist: {}", self.alist);
        if let Some(puncturing) = self.puncturing.as_ref() {
            println!(" - Puncturing pattern: {puncturing}");
        }
        println!(" - Information bits (k): {}", test.k());
        println!(" - Codeword size (N_cw): {}", test.n_cw());
        println!(" - Frame size (N): {}", test.n());
        println!(" - Code rate: {:.3}", test.rate());
        println!("LDPC decoder:");
        println!(" - Implementation: {}", self.decoder);
        println!(" - Maximum iterations: {}", self.max_iter);
        println!();
    }
}

fn parse_puncturing_pattern(s: &str) -> Result<Vec<bool>, &'static str> {
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
}

impl Progress {
    fn new(rx: Receiver<Report>) -> Progress {
        Progress {
            rx,
            term: Term::stdout(),
        }
    }

    fn run(&self) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
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

    fn work(&self) -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
        self.term.set_title("ldpc-toolbox ber");
        self.term.hide_cursor()?;
        self.term.write_line(Self::format_header())?;
        let mut current_ebn0 = None;
        loop {
            let Report::Statistics(stats) = self.rx.recv().unwrap() else {
                // BER test has finished
                return Ok(())
            };
            match current_ebn0 {
                Some(ebn0) if ebn0 == stats.ebn0_db => {
                    self.term.move_cursor_up(1)?;
                    self.term.clear_line()?;
                }
                _ => (),
            };
            current_ebn0 = Some(stats.ebn0_db);
            self.term.write_line(&Self::format_progress(&stats))?;
        }
    }

    fn format_header() -> &'static str {
        "  Eb/N0 |   Frames | Bit errs | Frame er | False de |     BER |     FER | Throughp | Elapsed\n\
         --------|----------|----------|----------|----------|---------|---------|----------|----------"
    }

    fn format_progress(stats: &Statistics) -> String {
        format!(
            "{:7.2} | {:8} | {:8} | {:8} | {:8} | {:7.2e} | {:7.2e} | {:8.3} | {}",
            stats.ebn0_db,
            stats.num_frames,
            stats.bit_errors,
            stats.frame_errors,
            stats.false_decodes,
            stats.ber,
            stats.fer,
            stats.throughput_mbps,
            humantime::format_duration(Duration::from_secs(stats.elapsed.as_secs()))
        )
    }
}
