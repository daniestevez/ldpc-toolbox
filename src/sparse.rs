//! # Sparse binary matrix representation and functions
//!
//! This module implements a representation for sparse binary matrices based on
//! the alist format used to handle LDPC parity check matrices.

use std::borrow::Borrow;
use std::slice::Iter;

mod bfs;
mod girth;

pub use bfs::BFSResults;

/// A [`String`] with an description of the error.
pub type Error = String;
/// A [`Result`] type containing an error [`String`].
pub type Result<T> = std::result::Result<T, Error>;

/// A sparse binary matrix
///
/// The internal representation for this matrix is based on the alist format.
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct SparseMatrix {
    rows: Vec<Vec<usize>>,
    cols: Vec<Vec<usize>>,
}

impl SparseMatrix {
    /// Create a new sparse matrix of a given size
    ///
    /// The matrix is inizialized to the zero matrix.
    ///
    /// # Examples
    /// ```
    /// # use ldpc_toolbox::sparse::SparseMatrix;
    /// let h = SparseMatrix::new(10, 30);
    /// assert_eq!(h.num_rows(), 10);
    /// assert_eq!(h.num_cols(), 30);
    /// ```
    pub fn new(nrows: usize, ncols: usize) -> SparseMatrix {
        use std::iter::repeat_with;
        let rows = repeat_with(Vec::new).take(nrows).collect();
        let cols = repeat_with(Vec::new).take(ncols).collect();
        SparseMatrix { rows, cols }
    }

    /// Returns the number of rows of the matrix
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns of the matrix
    pub fn num_cols(&self) -> usize {
        self.cols.len()
    }

    /// Returns the row weight of `row`
    ///
    /// The row weight is defined as the number of entries equal to
    /// one in a particular row. Rows are indexed starting by zero.
    pub fn row_weight(&self, row: usize) -> usize {
        self.rows[row].len()
    }

    /// Returns the column weight of `column`
    ///
    /// The column weight is defined as the number of entries equal to
    /// one in a particular column. Columns are indexed starting by zero.
    pub fn col_weight(&self, col: usize) -> usize {
        self.cols[col].len()
    }

    /// Returns `true` if the entry corresponding to a particular
    /// row and column is a one
    pub fn contains(&self, row: usize, col: usize) -> bool {
        // typically columns are shorter, so we search in the column
        self.cols[col].contains(&row)
    }

    /// Inserts a one in a particular row and column
    ///
    /// # Examples
    /// ```
    /// # use ldpc_toolbox::sparse::SparseMatrix;
    /// let mut h = SparseMatrix::new(10, 30);
    /// assert!(!h.contains(3, 7));
    /// h.insert(3, 7);
    /// assert!(h.contains(3, 7));
    /// ```
    pub fn insert(&mut self, row: usize, col: usize) {
        self.rows[row].push(col);
        self.cols[col].push(row);
    }

    /// Inserts ones in particular columns of a row
    ///
    /// This effect is as calling `insert()` on each of the elements
    /// of the iterator `cols`.
    ///
    /// # Examples
    /// ```
    /// # use ldpc_toolbox::sparse::SparseMatrix;
    /// let mut h1 = SparseMatrix::new(10, 30);
    /// let mut h2 = SparseMatrix::new(10, 30);
    /// let c = vec![3, 7, 9];
    /// h1.insert_row(0, c.iter());
    /// for a in &c {
    ///     h2.insert(0, *a);
    /// }
    /// assert_eq!(h1, h2);
    /// ```
    pub fn insert_row<T, S>(&mut self, row: usize, cols: T)
    where
        T: Iterator<Item = S>,
        S: Borrow<usize>,
    {
        for col in cols {
            self.insert(row, *col.borrow());
        }
    }

    /// Inserts ones in a particular rows of a column
    ///
    /// This works like `insert_row()`.
    pub fn insert_col<T, S>(&mut self, col: usize, rows: T)
    where
        T: Iterator<Item = S>,
        S: Borrow<usize>,
    {
        for row in rows {
            self.insert(*row.borrow(), col);
        }
    }

    /// Remove all the ones in a particular row
    pub fn clear_row(&mut self, row: usize) {
        for &col in &self.rows[row] {
            self.cols[col].retain(|r| *r != row);
        }
        self.rows[row].clear();
    }

    /// Remove all the ones in a particular column
    pub fn clear_col(&mut self, col: usize) {
        for &row in &self.cols[col] {
            self.rows[row].retain(|c| *c != col);
        }
        self.cols[col].clear();
    }

    /// Set the elements that are equal to one in a row
    ///
    /// The effect of this is like calling `clear_row()` followed
    /// by `insert_row()`.
    pub fn set_row<T, S>(&mut self, row: usize, cols: T)
    where
        T: Iterator<Item = S>,
        S: Borrow<usize>,
    {
        self.clear_row(row);
        self.insert_row(row, cols);
    }

    /// Set the elements that are equal to one in a column
    pub fn set_col<T, S>(&mut self, col: usize, rows: T)
    where
        T: Iterator<Item = S>,
        S: Borrow<usize>,
    {
        self.clear_col(col);
        self.insert_col(col, rows);
    }

    /// Returns an [Iterator] over the entries equal to one
    /// in a particular row
    pub fn iter_row(&self, row: usize) -> Iter<'_, usize> {
        self.rows[row].iter()
    }

    /// Returns an [Iterator] over the entries equal to one
    /// in a particular column
    pub fn iter_col(&self, col: usize) -> Iter<'_, usize> {
        self.cols[col].iter()
    }

    /// Writes the matrix in alist format to a writer
    ///
    /// # Errors
    /// If a call to `write!()` returns an error, this function returns
    /// such an error.
    pub fn write_alist<W: std::fmt::Write>(&self, w: &mut W) -> std::fmt::Result {
        writeln!(w, "{} {}", self.num_cols(), self.num_rows())?;
        let directions = [&self.cols, &self.rows];
        for dir in directions.iter() {
            write!(w, "{} ", dir.iter().map(|el| el.len()).max().unwrap_or(0))?;
        }
        writeln!(w)?;
        for dir in directions.iter() {
            for el in *dir {
                write!(w, "{} ", el.len())?;
            }
            writeln!(w)?;
        }
        for dir in directions.iter() {
            for el in *dir {
                let mut v = el.clone();
                v.sort_unstable();
                for x in &v {
                    write!(w, "{} ", x + 1)?;
                }
                writeln!(w)?;
            }
        }
        Ok(())
    }

    /// Returns a [`String`] with the alist representation of the matrix
    pub fn alist(&self) -> String {
        let mut s = String::new();
        self.write_alist(&mut s).unwrap();
        s
    }

    /// Constructs and returns a sparse matrix from its alist representation
    ///
    /// # Errors
    /// `alist` should hold a valid alist representation. If an error is found
    /// while parsing `alist`, a `String` describing the error will be returned.
    pub fn from_alist(alist: &str) -> Result<SparseMatrix> {
        let mut alist = alist.split('\n');
        let sizes = alist
            .next()
            .ok_or_else(|| String::from("alist first line not found"))?;
        let mut sizes = sizes.split_whitespace();
        let ncols = sizes
            .next()
            .ok_or_else(|| String::from("alist first line does not contain enough elements"))?
            .parse()
            .map_err(|_| String::from("ncols is not a number"))?;
        let nrows = sizes
            .next()
            .ok_or_else(|| String::from("alist first line does not contain enough elements"))?
            .parse()
            .map_err(|_| String::from("nrows is not a number"))?;
        let mut h = SparseMatrix::new(nrows, ncols);
        alist.next(); // skip max weights
        alist.next();
        alist.next(); // skip weights
        for col in 0..ncols {
            let col_data = alist
                .next()
                .ok_or_else(|| String::from("alist does not contain expected number of lines"))?;
            let col_data = col_data.split_whitespace();
            for row in col_data {
                let row: usize = row
                    .parse()
                    .map_err(|_| String::from("row value is not a number"))?;
                h.insert(row - 1, col);
            }
        }
        // we do not need to process the rows of the alist
        Ok(h)
    }

    /// Returns the girth of the bipartite graph defined by the matrix
    ///
    /// The girth is the length of the shortest cycle. If there are no
    /// cycles, `None` is returned.
    ///
    /// # Examples
    /// The following shows that a 2 x 2 matrix whose entries are all
    /// equal to one has a girth of 4, which is the smallest girth that
    /// a bipartite graph can have.
    /// ```
    /// # use ldpc_toolbox::sparse::SparseMatrix;
    /// let mut h = SparseMatrix::new(2, 2);
    /// for j in 0..2 {
    ///     for k in 0..2 {
    ///         h.insert(j, k);
    ///     }
    /// }
    /// assert_eq!(h.girth(), Some(4));
    /// ```
    pub fn girth(&self) -> Option<usize> {
        self.girth_with_max(usize::MAX)
    }

    /// Returns the girth of the bipartite graph defined by the matrix
    /// as long as it is smaller than a maximum
    ///
    /// By imposing a maximum value in the girth search algorithm,
    /// the execution time is reduced, since paths longer than the
    /// maximum do not need to be explored.
    ///
    /// Often it is only necessary to check that a graph has at least
    /// some minimum girth, so it is possible to use `girth_with_max()`.
    ///
    /// If there are no cycles with length smaller or equal to `max`, then
    /// `None` is returned.
    pub fn girth_with_max(&self, max: usize) -> Option<usize> {
        (0..self.num_cols())
            .filter_map(|c| self.girth_at_col_with_max(c, max))
            .min()
    }

    /// Returns the girth at a particular column
    ///
    /// The local girth at a node of a graph is defined as the minimum
    /// length of the cycles containing that node.
    ///
    /// This function returns the local girth of at the node correponding
    /// to a column of the matrix, or `None` if there are no cycles containing
    /// that node.
    pub fn girth_at_col(&self, col: usize) -> Option<usize> {
        self.girth_at_node(Node::Col(col))
    }

    /// Returns the girth at a particular column with a maximum
    ///
    /// This function works like `girth_at_col()` but imposes a maximum in the
    /// length of the cycles considered. `None` is returned if there are no
    /// cycles containing the node with length smaller or equal than `max`.
    pub fn girth_at_col_with_max(&self, col: usize, max: usize) -> Option<usize> {
        self.girth_at_node_with_max(Node::Col(col), max)
    }

    /// Returns the girth at a particular row
    ///
    /// This function works like `girth_at_col()` but uses the node
    /// corresponding to a row instead.
    pub fn girth_at_row(&self, row: usize) -> Option<usize> {
        self.girth_at_node(Node::Row(row))
    }

    /// Returns the girth at a particular row with a maximum
    ///
    /// This function works like `girth_at_col_with_max()` but uses the node
    /// corresponding to a row instead.
    pub fn girth_at_row_with_max(&self, row: usize, max: usize) -> Option<usize> {
        self.girth_at_node_with_max(Node::Row(row), max)
    }

    /// Returns the girth at a particular node
    ///
    /// This function works like `girth_at_col()` and
    /// `girth_at_row()`, but uses a `Node` to allow specifying either
    /// a column or a row.
    pub fn girth_at_node(&self, node: Node) -> Option<usize> {
        self.girth_at_node_with_max(node, usize::MAX)
    }

    /// Returns the girth at a particular node with a maximum
    ///
    /// This function works like `girth_at_col_with_max()` and
    /// `girth_at_row_with_max()`, but uses a `Node` to allow specifying either
    /// a column or a row.
    pub fn girth_at_node_with_max(&self, node: Node, max: usize) -> Option<usize> {
        bfs::BFSContext::new(self, node).local_girth(max)
    }

    pub fn bfs(&self, node: Node) -> BFSResults {
        bfs::BFSContext::new(self, node).bfs()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Node {
    Row(usize),
    Col(usize),
}

impl Node {
    fn iter(self, h: &SparseMatrix) -> impl Iterator<Item = Node> + '_ {
        match self {
            Node::Row(n) => h.iter_row(n),
            Node::Col(n) => h.iter_col(n),
        }
        .map(move |&x| match self {
            Node::Row(_) => Node::Col(x),
            Node::Col(_) => Node::Row(x),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut h = SparseMatrix::new(100, 300);
        assert!(!h.contains(27, 154));
        h.insert(27, 154);
        assert!(h.contains(27, 154));
        assert!(!h.contains(28, 154));
    }

    #[test]
    fn test_alist() {
        let mut h = SparseMatrix::new(4, 12);
        for j in 0..4 {
            h.insert(j, j);
            h.insert(j, j + 4);
            h.insert(j, j + 8);
        }
        let expected = "12 4
1 3 
1 1 1 1 1 1 1 1 1 1 1 1 
3 3 3 3 
1 
2 
3 
4 
1 
2 
3 
4 
1 
2 
3 
4 
1 5 9 
2 6 10 
3 7 11 
4 8 12 
";
        assert_eq!(h.alist(), expected);

        let h2 = SparseMatrix::from_alist(expected).unwrap();
        assert_eq!(h2.alist(), expected);
    }
}
