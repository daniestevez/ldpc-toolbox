use crate::sparse::SparseMatrix;

pub fn is_staircase(h: &SparseMatrix) -> bool {
    let n = h.num_rows();
    let m = h.num_cols();
    let mut num_checked = 0; // number of ones in parity part
                             // Check that all the ones in the parity part of the matrix are staircase
                             // positions
    for (j, k) in h.iter_all() {
        if k >= m - n {
            if j == 0 && k != m - n {
                // unexpected one in first row
                return false;
            }
            if j != 0 && k != m - n + j - 1 && k != m - n + j {
                // unexpected one in row other than first
                return false;
            }
            num_checked += 1;
        }
    }
    // There must be exactly 2n-1 ones in the staircase
    num_checked == 2 * n - 1
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn staircase() {
        let mut h = SparseMatrix::new(3, 5);
        assert!(!is_staircase(&h));
        h.insert(0, 2);
        assert!(!is_staircase(&h));
        h.insert(1, 2);
        assert!(!is_staircase(&h));
        h.insert(1, 3);
        assert!(!is_staircase(&h));
        h.insert(2, 3);
        assert!(!is_staircase(&h));
        h.insert(2, 4);
        assert!(is_staircase(&h)); // now it must be a staircase
        h.insert(0, 3);
        assert!(!is_staircase(&h)); // not a staircase anymore
    }
}
