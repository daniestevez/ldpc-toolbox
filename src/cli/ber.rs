//! BER test CLI subcommand.
//!
//! This subcommand can be used to perform a BER test of an LDPC decoder.

use crate::{cli::*, simulation::ber::BerTest, sparse::SparseMatrix};
use clap::Parser;
use std::error::Error;

/// BER test CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Performs a BER simulation")]
pub struct Args {
    /// alist file for the code
    alist: String,
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
        let test = BerTest::new(
            h,
            puncturing_pattern.as_ref().map(|v| &v[..]),
            self.frame_errors,
            self.max_iter,
            &ebn0s,
        )?;
        let mut rng = rand::thread_rng();
        let stats = test.run(&mut rng)?;
        println!("{:?}", stats);
        Ok(())
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
