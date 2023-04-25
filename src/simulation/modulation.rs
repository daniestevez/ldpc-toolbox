//! Modulation and demodulation.
//!
//! This module implements routines for modulation of bits to symbols and
//! demodulation of symbols to LLRs.

use crate::gf2::GF2;
use ndarray::{ArrayBase, Data, Ix1};
use num_traits::{One, Zero};

/// BPSK modulator.
///
/// Maps the bit 0 to the symbol -1.0 and the bit 1 to the symbol +1.0.
#[derive(Debug, Clone, Default)]
pub struct BpskModulator {}

impl BpskModulator {
    /// Creates a new BPSK modulator.
    pub fn new() -> BpskModulator {
        BpskModulator::default()
    }

    /// Modulates a sequence of bits into symbols.
    pub fn modulate<S>(&self, codeword: &ArrayBase<S, Ix1>) -> Vec<f64>
    where
        S: Data<Elem = GF2>,
    {
        codeword.iter().cloned().map(Self::modulate_bit).collect()
    }

    fn modulate_bit(bit: GF2) -> f64 {
        if bit.is_zero() {
            -1.0
        } else if bit.is_one() {
            1.0
        } else {
            panic!("invalid GF2 value")
        }
    }
}

/// BPSK demodulator.
///
/// Assumes the same mapping as the [BpskModulator].
#[derive(Debug, Clone, Default)]
pub struct BpskDemodulator {
    scale: f64,
}

impl BpskDemodulator {
    /// Creates a new BPSK demodulator.
    ///
    /// The `noise_sigma` indicates the channel noise standard deviation. The
    /// channel noise is assumed to be a real Gaussian with mean zero and
    /// standard deviation `noise_sigma`.
    pub fn new(noise_sigma: f64) -> BpskDemodulator {
        BpskDemodulator {
            // Negative scale because we use the convention that +1 means a 1
            // bit.
            scale: -2.0 / (noise_sigma * noise_sigma),
        }
    }

    /// Returns the LLRs corresponding to a sequence of symbols.
    pub fn demodulate(&self, symbols: &[f64]) -> Vec<f64> {
        symbols.iter().map(|&x| self.scale * x).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bpsk() {
        let modulator = BpskModulator::new();
        let x = modulator.modulate(&ndarray::arr1(&[GF2::one(), GF2::zero()]));
        assert_eq!(&x, &[1.0, -1.0]);
    }

    #[test]
    fn demodulator() {
        let demodulator = BpskDemodulator::new(2.0_f64.sqrt());
        let x = demodulator.demodulate(&[1.0, -1.0]);
        assert_eq!(x.len(), 2);
        let tol = 1e-4;
        assert!((x[0] + 1.0).abs() < tol);
        assert!((x[1] - 1.0).abs() < tol);
    }
}
