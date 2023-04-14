//! Finite field GF(2) arithmetic.
//!
//! This module contains the struct [GF2], which implements the finite field
//! arithmetic in GF(2).

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use ndarray::ScalarOperand;
use num_traits::{One, Zero};

/// Finite field GF(2) element.
///
/// This struct represents an element of the finite field GF(2).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct GF2(u8);

impl Zero for GF2 {
    fn zero() -> GF2 {
        GF2(0)
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    fn set_zero(&mut self) {
        *self = Self::zero()
    }
}

impl One for GF2 {
    fn one() -> GF2 {
        GF2(1)
    }

    fn set_one(&mut self) {
        *self = Self::one()
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

impl Add for GF2 {
    type Output = GF2;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: GF2) -> GF2 {
        GF2(self.0 ^ rhs.0)
    }
}

impl Sub for GF2 {
    type Output = GF2;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: GF2) -> GF2 {
        self + rhs
    }
}

impl Mul for GF2 {
    type Output = GF2;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: GF2) -> GF2 {
        GF2(self.0 & rhs.0)
    }
}

impl Div for GF2 {
    type Output = GF2;

    fn div(self, rhs: GF2) -> GF2 {
        if rhs.is_zero() {
            panic!("division by zero");
        }
        self
    }
}

macro_rules! impl_ops {
    ($op:ident, $opmethod:ident, $opassign:ident, $opassign_method:ident) => {
        impl $op<&GF2> for GF2 {
            type Output = GF2;
            fn $opmethod(self, rhs: &GF2) -> GF2 {
                self.$opmethod(*rhs)
            }
        }

        impl $opassign for GF2 {
            fn $opassign_method(&mut self, rhs: GF2) {
                *self = self.$opmethod(rhs);
            }
        }

        impl $opassign<&GF2> for GF2 {
            fn $opassign_method(&mut self, rhs: &GF2) {
                *self = self.$opmethod(*rhs);
            }
        }
    };
}

impl_ops!(Add, add, AddAssign, add_assign);
impl_ops!(Sub, sub, SubAssign, sub_assign);
impl_ops!(Mul, mul, MulAssign, mul_assign);
impl_ops!(Div, div, DivAssign, div_assign);

impl ScalarOperand for GF2 {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ops() {
        assert_eq!(GF2(0) + GF2(0), GF2(0));
        assert_eq!(GF2(0) + GF2(1), GF2(1));
        assert_eq!(GF2(1) + GF2(0), GF2(1));
        assert_eq!(GF2(1) + GF2(1), GF2(0));
        assert_eq!(GF2(0) - GF2(0), GF2(0));
        assert_eq!(GF2(0) - GF2(1), GF2(1));
        assert_eq!(GF2(1) - GF2(0), GF2(1));
        assert_eq!(GF2(1) - GF2(1), GF2(0));
        assert_eq!(GF2(0) * GF2(0), GF2(0));
        assert_eq!(GF2(0) * GF2(1), GF2(0));
        assert_eq!(GF2(1) * GF2(0), GF2(0));
        assert_eq!(GF2(1) * GF2(1), GF2(1));
        assert_eq!(GF2(0) / GF2(1), GF2(0));
        assert_eq!(GF2(1) / GF2(1), GF2(1));
    }

    #[test]
    #[should_panic]
    fn div_one_by_zero() {
        let _a = GF2(1) / GF2(0);
    }

    #[test]
    #[should_panic]
    fn div_zero_by_zero() {
        let _a = GF2(0) / GF2(0);
    }
}
