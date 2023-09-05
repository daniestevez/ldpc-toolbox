//! Modulation and demodulation.
//!
//! This module implements routines for modulation of bits to symbols and
//! demodulation of symbols to LLRs.

use super::channel::ChannelType;
use crate::gf2::GF2;
use ndarray::{ArrayBase, Data, Ix1};
use num_complex::Complex;
use num_traits::{One, Zero};

/// Modulation.
///
/// This trait is used to define the modulations that can be handled by the
/// simulation. It ties together a modulator and demodulator that work over the
/// same channel type (either real or complex), and declares the number of bits
/// per symbol of the modulation.
pub trait Modulation: 'static {
    /// Channel type.
    ///
    /// This is the scalar type for the symbols of the channel.
    type T: ChannelType;
    /// Modulator type.
    type Modulator: Modulator<T = Self::T>;
    /// Demodulator type.
    type Demodulator: Demodulator<T = Self::T>;
    /// Number of bits per symbol.
    const BITS_PER_SYMBOL: f64;
}

/// Modulator.
///
/// This trait defines modulators, which can convert a sequence of bits into
/// symbols.
pub trait Modulator: Default + Clone + Send {
    /// Scalar type for the symbols.
    type T;

    /// Modulates a sequence of bits into symbols.
    fn modulate<S>(&self, codeword: &ArrayBase<S, Ix1>) -> Vec<Self::T>
    where
        S: Data<Elem = GF2>;
}

/// Demodulator.
///
/// This trait defines demodulators, which can compute the bit LLRs for a
/// sequence of symbols.
pub trait Demodulator: Send {
    /// Scalar type for the symbols.
    type T;

    /// Creates a new demodulator.
    ///
    /// The parameter `noise_sigma` indicates the channel noise standard
    /// deviation in its real and imaginary part (or the channel noise standard
    /// deviation if the channel is real).
    fn from_noise_sigma(noise_sigma: f64) -> Self;

    /// Returns the LLRs corresponding to a sequence of symbols.
    fn demodulate(&self, symbols: &[Self::T]) -> Vec<f64>;
}

/// BPSK modulation
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct Bpsk {}

impl Modulation for Bpsk {
    type T = f64;
    type Modulator = BpskModulator;
    type Demodulator = BpskDemodulator;
    const BITS_PER_SYMBOL: f64 = 1.0;
}

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

impl Modulator for BpskModulator {
    type T = f64;

    fn modulate<S>(&self, codeword: &ArrayBase<S, Ix1>) -> Vec<f64>
    where
        S: Data<Elem = GF2>,
    {
        codeword.iter().cloned().map(Self::modulate_bit).collect()
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
}

impl Demodulator for BpskDemodulator {
    type T = f64;

    fn from_noise_sigma(noise_sigma: f64) -> BpskDemodulator {
        BpskDemodulator::new(noise_sigma)
    }

    fn demodulate(&self, symbols: &[f64]) -> Vec<f64> {
        symbols.iter().map(|&x| self.scale * x).collect()
    }
}

/// BPSK modulation
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub struct Psk8 {}

impl Modulation for Psk8 {
    type T = Complex<f64>;
    type Modulator = Psk8Modulator;
    type Demodulator = Psk8Demodulator;
    const BITS_PER_SYMBOL: f64 = 3.0;
}

/// 8PSK modulator.
///
/// 8PSK modulator using the DVB-S2 Gray-coded constellation. The modulator can
/// only work with codewords whose length is a multiple of 3 bits.
#[derive(Debug, Clone, Default)]
pub struct Psk8Modulator {}

impl Psk8Modulator {
    /// Creates a new 8PSK modulator.
    pub fn new() -> Psk8Modulator {
        Psk8Modulator::default()
    }

    fn modulate_bits(b0: GF2, b1: GF2, b2: GF2) -> Complex<f64> {
        let a = (0.5f64).sqrt();
        match (b0.is_one(), b1.is_one(), b2.is_one()) {
            (false, false, false) => Complex::new(a, a),
            (true, false, false) => Complex::new(0.0, 1.0),
            (true, true, false) => Complex::new(-a, a),
            (false, true, false) => Complex::new(-1.0, 0.0),
            (false, true, true) => Complex::new(-a, -a),
            (true, true, true) => Complex::new(0.0, -1.0),
            (true, false, true) => Complex::new(a, -a),
            (false, false, true) => Complex::new(1.0, 0.0),
        }
    }
}

impl Modulator for Psk8Modulator {
    type T = Complex<f64>;

    /// Modulates a sequence of bits into symbols.
    ///
    /// # Panics
    ///
    /// Panics if the length of the codeword is not a multiple of 3 bits.
    fn modulate<S>(&self, codeword: &ArrayBase<S, Ix1>) -> Vec<Complex<f64>>
    where
        S: Data<Elem = GF2>,
    {
        assert_eq!(codeword.len() % 3, 0);
        codeword
            .iter()
            .step_by(3)
            .zip(codeword.iter().skip(1).step_by(3))
            .zip(codeword.iter().skip(2).step_by(3))
            .map(|((&b0, &b1), &b2)| Self::modulate_bits(b0, b1, b2))
            .collect()
    }
}

/// 8PSK demodulator.
///
/// Assumes the same mapping as the [Psk8Modulator]. Demodulates symbols into
/// LLRs using the exact formula implemented with the max-* function.
#[derive(Debug, Clone, Default)]
pub struct Psk8Demodulator {
    scale: f64,
}

impl Psk8Demodulator {
    /// Creates a new 8PSK demodulator.
    ///
    /// The `noise_sigma` indicates the channel noise standard deviation. The
    /// channel noise is assumed to be a circularly symmetric Gaussian with mean
    /// zero and standard deviation `noise_sigma` in its real part and imaginary
    /// part (the total variance is `2 * noise_sigma * noise_sigma`.
    pub fn new(noise_sigma: f64) -> Psk8Demodulator {
        Psk8Demodulator {
            scale: 1.0 / (noise_sigma * noise_sigma),
        }
    }

    fn demodulate_symbol(&self, symbol: Complex<f64>) -> [f64; 3] {
        let a = (0.5f64).sqrt();
        let symbol = symbol * self.scale;
        let d000 = dot(symbol, Complex::new(a, a));
        let d100 = dot(symbol, Complex::new(0.0, 1.0));
        let d110 = dot(symbol, Complex::new(-a, a));
        let d010 = dot(symbol, Complex::new(-1.0, 0.0));
        let d011 = dot(symbol, Complex::new(-a, -a));
        let d111 = dot(symbol, Complex::new(0.0, -1.0));
        let d101 = dot(symbol, Complex::new(a, -a));
        let d001 = dot(symbol, Complex::new(1.0, 0.0));
        let b0 = [d000, d001, d010, d011]
            .into_iter()
            .reduce(maxstar)
            .unwrap()
            - [d100, d101, d110, d111]
                .into_iter()
                .reduce(maxstar)
                .unwrap();
        let b1 = [d000, d001, d100, d101]
            .into_iter()
            .reduce(maxstar)
            .unwrap()
            - [d010, d011, d110, d111]
                .into_iter()
                .reduce(maxstar)
                .unwrap();
        let b2 = [d000, d010, d100, d110]
            .into_iter()
            .reduce(maxstar)
            .unwrap()
            - [d001, d011, d101, d111]
                .into_iter()
                .reduce(maxstar)
                .unwrap();
        [b0, b1, b2]
    }
}

impl Demodulator for Psk8Demodulator {
    type T = Complex<f64>;

    fn from_noise_sigma(noise_sigma: f64) -> Psk8Demodulator {
        Psk8Demodulator::new(noise_sigma)
    }

    fn demodulate(&self, symbols: &[Complex<f64>]) -> Vec<f64> {
        symbols
            .iter()
            .flat_map(|&x| self.demodulate_symbol(x))
            .collect()
    }
}

fn dot(a: Complex<f64>, b: Complex<f64>) -> f64 {
    a.re * b.re + a.im * b.im
}

fn maxstar(a: f64, b: f64) -> f64 {
    a.max(b) + (-((a - b).abs())).exp().ln_1p()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bpsk_modulator() {
        let modulator = BpskModulator::new();
        let x = modulator.modulate(&ndarray::arr1(&[GF2::one(), GF2::zero()]));
        assert_eq!(&x, &[1.0, -1.0]);
    }

    #[test]
    fn bpsk_demodulator() {
        let demodulator = BpskDemodulator::new(2.0_f64.sqrt());
        let x = demodulator.demodulate(&[1.0, -1.0]);
        assert_eq!(x.len(), 2);
        let tol = 1e-4;
        assert!((x[0] + 1.0).abs() < tol);
        assert!((x[1] - 1.0).abs() < tol);
    }

    #[test]
    fn psk8_modulator() {
        let o = GF2::one();
        let z = GF2::zero();
        let modulator = Psk8Modulator::new();
        let x = modulator.modulate(&ndarray::arr1(&[o, o, z, z, z, z, o, z, o]));
        let a = (0.5f64).sqrt();
        assert_eq!(
            &x,
            &[Complex::new(-a, a), Complex::new(a, a), Complex::new(a, -a)]
        );
    }

    #[test]
    fn psk8_demodulator_signs() {
        let noise_sigma = 1.0;
        let demodulator = Psk8Demodulator::new(noise_sigma);
        let a = (0.5f64).sqrt();
        let llr = demodulator.demodulate(&[
            Complex::new(1.0, 0.0),
            Complex::new(a, a),
            Complex::new(0.0, 1.0),
        ]);
        // 001
        assert!(llr[0] > 0.0);
        assert!(llr[1] > 0.0);
        assert!(llr[2] < 0.0);
        // 000
        assert!(llr[3] > 0.0);
        assert!(llr[4] > 0.0);
        assert!(llr[5] > 0.0);
        // 100
        assert!(llr[6] < 0.0);
        assert!(llr[7] > 0.0);
        assert!(llr[8] > 0.0);
    }
}
