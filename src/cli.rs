//! Implementation of the CLI application of ldpc-toolbox

use std::error::Error;
use structopt::StructOpt;

pub mod dvbs2;
pub mod mackay_neal;
pub mod peg;

/// Trait to run a CLI subcommand
pub trait Run {
    /// Run the CLI subcommand
    fn run(&self) -> Result<(), Box<dyn Error>>;
}

/// CLI options
#[derive(Debug, StructOpt)]
#[structopt(name = "ldpc-toolbox", about = "LDPC toolbox")]
pub enum Opt {
    /// dvbs2 subcommand
    DVBS2(dvbs2::Opt),
    /// mackay-neal subcommand
    MackayNeal(mackay_neal::Opt),
    /// peg subcommand
    PEG(peg::Opt),
}

impl Run for Opt {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        match self {
            Opt::DVBS2(x) => x.run(),
            Opt::MackayNeal(x) => x.run(),
            Opt::PEG(x) => x.run(),
        }
    }
}
