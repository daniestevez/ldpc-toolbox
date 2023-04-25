//! Channel simulation.
//!
//! This module contains the simulation of an AWGN channel.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// AWGN channel simulation.
///
/// This struct is used to add AWGN to symbols.
#[derive(Debug, Clone)]
pub struct AwgnChannel {
    distr: Normal<f64>,
}

impl AwgnChannel {
    /// Creates a new AWGN channel.
    ///
    /// The channel noise follows a (real) normal distribution with mean zero
    /// and standard deviation sigma.
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

    /// Adds noise to a sequence of symbols.
    ///
    /// The noise is added in-place to the slice `symbols`. An [Rng] is used as
    /// source of randomness.
    pub fn add_noise<R: Rng>(&self, rng: &mut R, symbols: &mut [f64]) {
        for x in symbols.iter_mut() {
            *x += self.distr.sample(rng);
        }
    }
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
        let mut rng = rand::thread_rng();
        let mut symbols = vec![1.0; 1024];
        let symbols_orig = symbols.clone();
        channel.add_noise(&mut rng, &mut symbols);
        assert_eq!(&symbols, &symbols_orig);
    }
}
