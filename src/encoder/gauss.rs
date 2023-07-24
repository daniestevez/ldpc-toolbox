use ndarray::{s, Array2, LinalgScalar};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Error {
    NotInvertible,
}

pub fn gauss_reduction<A: LinalgScalar + PartialEq>(array: &mut Array2<A>) -> Result<(), Error> {
    let (n, m) = array.dim();

    // Reduce to upper triangular with ones on diagonal
    for j in 0..n {
        // Find non-zero element in current column
        let Some(k) = array.slice(s![j.., j]).iter().enumerate().find_map(|(t, x)| if x.is_zero() {
            None
        } else {
            Some(j + t)
        }) else {
            return Err(Error::NotInvertible);
        };

        if k != j {
            // Swap rows j and k
            for t in j..m {
                array.swap([j, t], [k, t]);
            }
        }

        // Make a 1 by dividing
        let x = array[[j, j]];
        if !x.is_one() {
            for t in j..m {
                array[[j, t]] = array[[j, t]] / x;
            }
        }

        // Subtract to rows below to make zeros below diagonal
        for t in (j + 1)..n {
            let x = array[[t, j]];
            if !x.is_zero() {
                // avoid calculations if we're subtracting zero
                for u in j..m {
                    array[[t, u]] = array[[t, u]] - x * array[[j, u]];
                }
            }
        }
    }

    // Reduce to identity
    for j in (0..n).rev() {
        // Subtract to rows above to make zeros above diagonal
        for t in 0..j {
            let x = array[[t, j]];
            if !x.is_zero() {
                // avoid calculations if we're subtracting zero
                for u in j..m {
                    array[[t, u]] = array[[t, u]] - x * array[[j, u]];
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gf2::GF2;
    use ndarray::arr2;
    use num_traits::{One, Zero};

    #[test]
    fn gauss() {
        let i = GF2::one();
        let o = GF2::zero();
        let mut a = arr2(&[
            [i, o, i, i, i, o, i, o, i],
            [i, i, o, o, i, i, o, i, o],
            [i, i, i, o, o, i, i, o, i],
        ]);
        gauss_reduction(&mut a).unwrap();
        let expected = arr2(&[
            [i, o, o, i, o, o, o, i, o],
            [o, i, o, i, i, i, o, o, o],
            [o, o, i, o, i, o, i, i, i],
        ]);
        assert_eq!(&a, &expected);
    }
}
