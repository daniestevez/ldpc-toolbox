use clap::{Parser, ValueEnum, builder::PossibleValue};
use ldpc_toolbox::{
    cli::{Run, ber::Args},
    decoder::{
        DecoderOutput, LdpcDecoder,
        factory::{self, DecoderFactory},
    },
    sparse::SparseMatrix,
};
use std::{error::Error, fmt::Display, sync::LazyLock};

#[derive(Debug)]
struct ExampleDecoder {}

impl LdpcDecoder for ExampleDecoder {
    fn decode(
        &mut self,
        llrs: &[f64],
        _max_iterations: usize,
    ) -> Result<DecoderOutput, DecoderOutput> {
        // This example decoder immediately fails, returning the hard decision
        // of the unmodified codeword
        let codeword = llrs.iter().map(|&x| if x > 0.0 { 1 } else { 0 }).collect();
        Err(DecoderOutput {
            codeword,
            iterations: 0,
        })
    }
}

// This enum extends ldpc_toolbox's DecoderFactory to include the example decoder
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum DecoderImplementation {
    DecoderImplementation(factory::DecoderImplementation),
    Example,
}

impl DecoderFactory for DecoderImplementation {
    fn build_decoder(&self, h: SparseMatrix) -> Box<dyn LdpcDecoder> {
        match self {
            DecoderImplementation::DecoderImplementation(d) => d.build_decoder(h),
            DecoderImplementation::Example => Box::new(ExampleDecoder {}),
        }
    }
}

impl Display for DecoderImplementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            DecoderImplementation::DecoderImplementation(d) => d.fmt(f),
            DecoderImplementation::Example => write!(f, "Example"),
        }
    }
}

impl ValueEnum for DecoderImplementation {
    fn value_variants<'a>() -> &'a [Self] {
        static VARIANTS: LazyLock<Vec<DecoderImplementation>> = LazyLock::new(|| {
            let mut variants = factory::DecoderImplementation::value_variants()
                .iter()
                .map(|&variant| DecoderImplementation::DecoderImplementation(variant))
                .collect::<Vec<_>>();
            variants.push(DecoderImplementation::Example);
            variants
        });
        &VARIANTS
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        match self {
            DecoderImplementation::DecoderImplementation(a) => a.to_possible_value(),
            DecoderImplementation::Example => {
                Some(PossibleValue::new("Example").help("Example decoder"))
            }
        }
    }
}

#[termination::display]
fn main() -> Result<(), Box<dyn Error>> {
    Args::<DecoderImplementation>::parse().run()
}
