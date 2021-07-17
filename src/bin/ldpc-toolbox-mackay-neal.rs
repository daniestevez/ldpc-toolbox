use ldpc_toolbox::mackay_neal::{Config, FillPolicy};
use std::error::Error;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "ldpc-toolbox-mackay-neal",
    about = "Generates LDPC codes using the MacKay-Neal algorithm"
)]
struct Opt {
    /// Number of rows
    num_rows: usize,
    /// Number of columns
    num_columns: usize,
    /// Maximum row weight
    wr: usize,
    /// Column weight
    wc: usize,
    /// Seed
    seed: u64,
    /// Columns to backtrack
    #[structopt(long, default_value = "0")]
    backtrack_cols: usize,
    /// Backtrack attemps
    #[structopt(long, default_value = "0")]
    backtrack_trials: usize,
    /// Minimum girth
    #[structopt(long)]
    min_girth: Option<usize>,
    /// Girth trials
    #[structopt(long, default_value = "0")]
    girth_trials: usize,
    /// Use uniform fill policy
    #[structopt(long)]
    uniform: bool,
    /// Maximum seed trials
    #[structopt(long, default_value = "1000")]
    seed_trials: u64,
}

impl Opt {
    fn config(&self) -> Config {
        Config {
            nrows: self.num_rows,
            ncols: self.num_columns,
            wr: self.wr,
            wc: self.wc,
            backtrack_cols: self.backtrack_cols,
            backtrack_trials: self.backtrack_trials,
            min_girth: self.min_girth,
            girth_trials: self.girth_trials,
            fill_policy: match self.uniform {
                true => FillPolicy::Uniform,
                false => FillPolicy::Random,
            },
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let opt = Opt::from_args();
    let conf = opt.config();
    let (seed, h) = conf.search(opt.seed, opt.seed_trials);
    eprintln!("seed = {}", seed);
    println!("{}", h.alist());
    Ok(())
}
