//! `ldpc-toolbox` CLI application
//!
//! The CLI application is organized in several subcommands. The
//! supported subcommands can be seen by running `ldpc-toolbox`.
//! See the modules below for examples and more information about
//! how to use each subcommand.

use clap::Parser;
use std::error::Error;

pub mod ber;
pub mod ccsds;
pub mod ccsds_c2;
pub mod ccsds_tc;
pub mod dvbs2;
pub mod encode;
pub mod mackay_neal;
pub mod peg;
pub mod systematic;

/// Trait to run a CLI subcommand
pub trait Run {
    /// Run the CLI subcommand
    fn run(&self) -> Result<(), Box<dyn Error>>;
}

/// CLI arguments.
#[derive(Debug, Parser)]
#[command(author, version, name = "ldpc-toolbox", about = "LDPC toolbox")]
pub enum Args {
    /// ber subcommand
    BER(ber::Args),
    /// ccsds subcommand
    CCSDS(ccsds::Args),
    /// ccsds-c2 subcommand
    #[allow(non_camel_case_types)]
    CCSDS_C2(ccsds_c2::Args),
    /// ccsds-tc subcommand
    #[allow(non_camel_case_types)]
    CCSDS_TC(ccsds_tc::Args),
    /// encode subcommand
    Encode(encode::Args),
    /// dvbs2 subcommand
    DVBS2(dvbs2::Args),
    /// mackay-neal subcommand
    MackayNeal(mackay_neal::Args),
    /// peg subcommand
    PEG(peg::Args),
    /// systematic subcommand
    Systematic(systematic::Args),
}

impl Run for Args {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        match self {
            Args::BER(x) => x.run(),
            Args::CCSDS(x) => x.run(),
            Args::CCSDS_C2(x) => x.run(),
            Args::CCSDS_TC(x) => x.run(),
            Args::DVBS2(x) => x.run(),
            Args::Encode(x) => x.run(),
            Args::MackayNeal(x) => x.run(),
            Args::PEG(x) => x.run(),
            Args::Systematic(x) => x.run(),
        }
    }
}
