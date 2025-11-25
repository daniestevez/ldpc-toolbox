//! Encode CLI subcommand.
//!
//! This command can be used to encode using a systematic LDPC code.

use super::ber::parse_puncturing_pattern;
use crate::{
    cli::Run, encoder::Encoder, gf2::GF2, simulation::puncturing::Puncturer, sparse::SparseMatrix,
};
use clap::Parser;
use ndarray::Array1;
use num_traits::{One, Zero};
use std::{
    error::Error,
    fs::File,
    io::{ErrorKind, Read, Write},
    path::PathBuf,
};

/// Encode CLI arguments.
#[derive(Debug, Parser)]
#[command(about = "Performs LDPC encoding")]
pub struct Args {
    /// alist file for the code
    pub alist: PathBuf,
    /// input file (information words as unpacked bits)
    pub input: PathBuf,
    /// output file (punctured words as unpacked bits)
    pub output: PathBuf,
    /// Puncturing pattern (format "1,1,1,0")
    #[structopt(long)]
    pub puncturing: Option<String>,
}

impl Run for Args {
    fn run(&self) -> Result<(), Box<dyn Error>> {
        let puncturer = if let Some(p) = self.puncturing.as_ref() {
            Some(Puncturer::new(&parse_puncturing_pattern(p)?))
        } else {
            None
        };
        let h = SparseMatrix::from_alist(&std::fs::read_to_string(&self.alist)?)?;
        let mut input = File::open(&self.input)?;
        let mut output = File::create(&self.output)?;
        let encoder = Encoder::from_h(&h)?;
        let n = h.num_cols();
        let k = n - h.num_rows();
        let mut information_word = vec![0; k];
        let mut codeword_buf = vec![0; n];
        loop {
            match input.read_exact(&mut information_word[..]) {
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                ret => ret?,
            };
            let word = Array1::from_iter(
                information_word
                    .iter()
                    .map(|&b| if b == 1 { GF2::one() } else { GF2::zero() }),
            );
            let codeword = encoder.encode(&word);
            let codeword = match &puncturer {
                Some(p) => p.puncture(&codeword)?,
                None => codeword,
            };
            for (x, y) in codeword.iter().zip(codeword_buf.iter_mut()) {
                *y = x.is_one().into();
            }
            output.write_all(&codeword_buf)?;
        }
        Ok(())
    }
}
