//! CCSDS C2 CLI subcommand
//!
//! This subcommand can be used to generate the C2 LDPC code (rate ~7/8)
//! described in the CCSDS TM Synchronization and Channel Coding Blue Book
//! standard. It will print the alist of the parity check matrix to
//! `stdout`. See [`crate::codes::ccsds`] for more information about the CCSDS
//! LDPC codes.
//!
//! # Examples
//! The parity check matrix can be generated with
//! ```shell
//! $ ldpc-toolbox ccsds-c2
//! ```

use crate::cli::*;
use crate::codes::ccsds::C2Code;
use clap::Parser;

/// CCSDS CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Generates the alist of CCSDS C2 LDPC")]
pub struct Args {}

impl Run for Args {
    fn run(&self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let h = C2Code::new().h();
        print!("{}", h.alist());
        Ok(())
    }
}
