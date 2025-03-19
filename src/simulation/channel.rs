//! Channel simulation.
//!
//! This module contains the simulation of an AWGN channel.

use num_complex::Complex;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Channel type.
///
/// Represents a real or complex (IQ) channel.
///
/// This trait is implemented for `f64` and `Complex<f64>` as a way of handling
/// both real and complex channels internally.
pub trait ChannelType: sealed::Sealed + std::ops::AddAssign + Sized {
    #[doc(hidden)]
    fn noise<R: Rng>(awgn_channel: &AwgnChannel, rng: &mut R) -> Self;
}

/// Channel model.
///
/// A channel model is able to add noise to a sequence of symbols, which can be
/// either real or complex.
pub trait Channel {
    /// Adds noise to a sequence of symbols.
    ///
    /// The noise is added in-place to the slice `symbols`. An [Rng] is used as
    /// source of randomness.
    fn add_noise<R: Rng, T: ChannelType>(&self, rng: &mut R, symbols: &mut [T]);
}

/// AWGN channel simulation.
///
/// This struct is used to add AWGN to symbols.
#[derive(Debug, Clone)]
pub struct AwgnChannel {
    distr: Normal<f64>,
}

impl AwgnChannel {
    /// Creates a new AWGN channel (either real or complex).
    ///
    /// When the channel is real, the channel noise follows a (real) normal
    /// distribution with mean zero and standard deviation `noise_sigma`. When
    /// the channel is complex, the channel noise follows a circularly symmetric
    /// normal distribution with mean zero and standard deviation of its real
    /// and imaginary part `noise_sigma`.
    ///
    /// # Panics
    ///
    /// This function panics if `noise_sigma` is not a positive finite number.
    pub fn new(noise_sigma: f64) -> AwgnChannel {
        assert!(noise_sigma >= 0.0);
        AwgnChannel {
            distr: Normal::new(0.0, noise_sigma).unwrap(),
        }
    }
}

impl Channel for AwgnChannel {
    fn add_noise<R: Rng, T: ChannelType>(&self, rng: &mut R, symbols: &mut [T]) {
        for x in symbols.iter_mut() {
            *x += T::noise(self, rng);
        }
    }
}

impl ChannelType for f64 {
    fn noise<R: Rng>(awgn_channel: &AwgnChannel, rng: &mut R) -> f64 {
        awgn_channel.distr.sample(rng)
    }
}

impl ChannelType for Complex<f64> {
    fn noise<R: Rng>(awgn_channel: &AwgnChannel, rng: &mut R) -> Complex<f64> {
        Complex::new(
            awgn_channel.distr.sample(rng),
            awgn_channel.distr.sample(rng),
        )
    }
}

mod sealed {
    use num_complex::Complex;
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for Complex<f64> {}
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn build_awgn() {
        let _channel = AwgnChannel::new(0.2);
    }

    #[test]
    #[should_panic]
    fn negative_noise_sigma() {
        let _channel = AwgnChannel::new(-3.5);
    }

    #[test]
    fn zero_noise_sigma() {
        let channel = AwgnChannel::new(0.0);
        let mut rng = rand::rng();
        let mut symbols = vec![1.0; 1024];
        let symbols_orig = symbols.clone();
        channel.add_noise(&mut rng, &mut symbols);
        assert_eq!(&symbols, &symbols_orig);
    }
}
