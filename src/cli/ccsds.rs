//! CCSDS CLI subcommand
//!
//! This subcommand can be used to generate the AR4JA LDPC codes described in
//! the CCSDS TM Synchronization and Channel Coding Blue Book standard. It will
//! print the alist of the parity check matrix to `stdout` and optionally
//! compute and print the girth of the Tanner graph. See [`crate::codes::ccsds`]
//! for more information about the CCSDS LDPC codes.
//!
//! # Examples
//! The r=1/2, k=1024 parity check matrix can be generated with
//! ```shell
//! $ ldpc-toolbox ccsds --rate 1/2 --block-size 1024
//! ```
//! Its girth is computed with
//! ```shell
//! $ ldpc-toolbox ccsds --rate 1/2 --block-size 1024 --girth
//! Code girth = 6
//! ```

use crate::cli::*;
use crate::codes::ccsds::{AR4JACode, AR4JAInfoSize, AR4JARate};
use clap::Parser;

type Error = String;
type Result<T> = std::result::Result<T, Error>;

/// CCSDS CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Generates the alist of CCSDS LDPCs")]
pub struct Args {
    /// Coding rate
    #[arg(short, long)]
    rate: String,

    /// Information block size (k)
    #[arg(long)]
    block_size: usize,

    /// Performs girth calculation
    #[arg(long)]
    girth: bool,
}

impl Args {
    fn code(&self) -> Result<AR4JACode> {
        let rate = match &*self.rate {
            "1/2" => AR4JARate::R1_2,
            "2/3" => AR4JARate::R2_3,
            "4/5" => AR4JARate::R4_5,
            r => return Err(format!("Invalid code rate {}", r)),
        };
        let info_size = match self.block_size {
            1024 => AR4JAInfoSize::K1024,
            4096 => AR4JAInfoSize::K4096,
            16384 => AR4JAInfoSize::K16384,
            s => return Err(format!("Invalid information block size k = {}", s)),
        };
        Ok(AR4JACode::new(rate, info_size))
    }
}

impl Run for Args {
    fn run(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let h = self.code()?.h();
        if self.girth {
            if let Some(g) = h.girth() {
                println!("Code girth = {}", g);
            } else {
                println!("Code girth is infinite");
            }
        } else {
            print!("{}", h.alist());
        }
        Ok(())
    }
}
