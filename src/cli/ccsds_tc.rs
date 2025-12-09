//! CCSDS TC CLI subcommand
//!
//! This subcommand can be used to generate the TC LDPC codes
//! described in the CCSDS TC Synchronization and Channel Coding Blue Book
//! standard. It will print the alist of the parity check matrix to
//! `stdout`. See [`crate::codes::ccsds_tc`] for more information about the CCSDS
//! TC codes.
//!
//! # Examples
//! The parity check matrix can be generated with
//! ```shell
//! $ ldpc-toolbox ccsds-tc
//! ```

use crate::cli::*;
use crate::codes::ccsds::{TCCode, TCInfoSize};
use clap::Parser;

type Error = String;
type Result<T> = std::result::Result<T, Error>;

/// CCSDS CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Generates the alist of CCSDS TC LDPC")]
pub struct Args {
    /// Information block size (k)
    #[arg(long)]
    block_size: usize,
}

impl Args {
    fn code(&self) -> Result<TCCode> {
        let info_size = match self.block_size {
            64 => TCInfoSize::K64,
            256 => TCInfoSize::K256,
            s => return Err(format!("Invalid information block size k = {}", s)),
        };
        Ok(TCCode::new(info_size))
    }
}

impl Run for Args {
    fn run(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let h = self.code()?.h();
        print!("{}", h.alist());
        Ok(())
    }
}
