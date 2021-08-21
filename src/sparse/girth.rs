#[cfg(test)]
mod tests {
    use crate::sparse::SparseMatrix;

    #[test]
    fn test_local_girth_circulant() {
        for circulant_size in 2..5 {
            let mut h = SparseMatrix::new(10, 20);
            for j in 0..circulant_size {
                h.insert(j, j);
                h.insert(j, (j + 1) % circulant_size);
            }
            for j in 0..circulant_size {
                assert_eq!(h.girth_at_col_with_max(j, 10), Some(2 * circulant_size));
                assert_eq!(h.girth_at_row_with_max(j, 10), Some(2 * circulant_size));
            }
            for j in circulant_size..h.num_cols() {
                assert_eq!(h.girth_at_col_with_max(j, 10), None);
            }
            for j in circulant_size..h.num_rows() {
                assert_eq!(h.girth_at_row_with_max(j, 10), None);
            }
        }
    }

    #[test]
    fn test_girth_identity() {
        let size = 20;
        let mut h = SparseMatrix::new(size, size);
        for j in 0..size {
            h.insert(j, j);
        }
        assert_eq!(h.girth_with_max(100), None);
    }

    #[test]
    fn test_girth_double_circulant() {
        let mut h = SparseMatrix::new(20, 30);
        let circulant_sizes = (5, 3);
        for j in 0..circulant_sizes.0 {
            h.insert(j, j);
            h.insert(j, (j + 1) % circulant_sizes.0);
        }
        for j in 0..circulant_sizes.1 {
            h.insert(j + circulant_sizes.0, j + circulant_sizes.0);
            h.insert(
                j + circulant_sizes.0,
                (j + 1) % circulant_sizes.1 + circulant_sizes.0,
            );
        }
        assert_eq!(h.girth_with_max(100), Some(2 * circulant_sizes.1));
        for j in 0..circulant_sizes.0 {
            assert_eq!(h.girth_at_col_with_max(j, 100), Some(2 * circulant_sizes.0));
            assert_eq!(h.girth_at_row_with_max(j, 100), Some(2 * circulant_sizes.0));
        }
        for j in circulant_sizes.0..(circulant_sizes.0 + circulant_sizes.1) {
            assert_eq!(h.girth_at_col_with_max(j, 100), Some(2 * circulant_sizes.1));
            assert_eq!(h.girth_at_row_with_max(j, 100), Some(2 * circulant_sizes.1));
        }
        for j in (circulant_sizes.0 + circulant_sizes.1)..h.num_cols() {
            assert_eq!(h.girth_at_col_with_max(j, 100), None);
        }
        for j in (circulant_sizes.0 + circulant_sizes.1)..h.num_rows() {
            assert_eq!(h.girth_at_row_with_max(j, 100), None);
        }
    }
}
