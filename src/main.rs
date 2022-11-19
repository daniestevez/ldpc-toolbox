use clap::Parser;
use ldpc_toolbox::cli::{Args, Run};
use std::error::Error;

#[termination::display]
fn main() -> Result<(), Box<dyn Error>> {
    Args::parse().run()
}
