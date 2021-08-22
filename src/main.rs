use ldpc_toolbox::cli::{Opt, *};
use std::error::Error;
use structopt::StructOpt;

#[termination::display]
fn main() -> Result<(), Box<dyn Error>> {
    Opt::from_args().run()
}
