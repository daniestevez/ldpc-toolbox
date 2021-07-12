use crate::sparse::SparseMatrix;
use std::collections::VecDeque;

enum Node {
    Row(usize),
    Col(usize),
}

struct PathHead {
    node: Node,
    parent: Option<usize>,
    length: usize,
}

impl PathHead {
    fn iter<'a>(&'a self, h: &'a SparseMatrix) -> impl Iterator<Item = PathHead> + 'a {
        let (n, it) = match self.node {
            Node::Row(n) => (n, h.iter_row(n)),
            Node::Col(n) => (n, h.iter_col(n)),
        };
        it.filter(move |&&x| {
            if let Some(parent) = self.parent {
                x != parent
            } else {
                true
            }
        })
        .map(move |&x| PathHead {
            node: match self.node {
                Node::Row(_) => Node::Col(x),
                Node::Col(_) => Node::Row(x),
            },
            parent: Some(n),
            length: self.length + 1,
        })
    }
}

struct BFSContext {
    row_nodes: Vec<Option<usize>>,
    col_nodes: Vec<Option<usize>>,
    to_visit: VecDeque<PathHead>,
}

impl BFSContext {
    fn new(nrows: usize, ncols: usize, node: Node) -> BFSContext {
        let mut to_visit = VecDeque::new();
        to_visit.push_back(PathHead {
            node,
            parent: None,
            length: 0,
        });
        BFSContext {
            row_nodes: std::iter::repeat(None).take(nrows).collect(),
            col_nodes: std::iter::repeat(None).take(ncols).collect(),
            to_visit,
        }
    }

    fn from_matrix(h: &SparseMatrix, node: Node) -> BFSContext {
        BFSContext::new(h.num_rows(), h.num_cols(), node)
    }

    fn get_node_mut(&mut self, node: &Node) -> &mut Option<usize> {
        match *node {
            Node::Row(n) => &mut self.row_nodes[n],
            Node::Col(n) => &mut self.col_nodes[n],
        }
    }
}

fn girth_at_node_with_max(h: &SparseMatrix, node: Node, max: usize) -> Option<usize> {
    let mut ctx = BFSContext::from_matrix(h, node);
    while let Some(head) = ctx.to_visit.pop_front() {
        for node in head.iter(h) {
            let target = ctx.get_node_mut(&node.node);
            if let Some(target_length) = *target {
                let total = target_length + node.length;
                return if total <= max { Some(total) } else { None };
            } else {
                *target = Some(node.length);
                if node.length < max {
                    ctx.to_visit.push_back(node);
                }
            }
        }
    }
    None
}

pub fn girth_at_col_with_max(h: &SparseMatrix, col: usize, max: usize) -> Option<usize> {
    girth_at_node_with_max(h, Node::Col(col), max)
}

pub fn girth_at_col(h: &SparseMatrix, col: usize) -> Option<usize> {
    girth_at_col_with_max(h, col, usize::MAX)
}

pub fn girth_at_row_with_max(h: &SparseMatrix, row: usize, max: usize) -> Option<usize> {
    girth_at_node_with_max(h, Node::Row(row), max)
}

pub fn girth_at_row(h: &SparseMatrix, row: usize) -> Option<usize> {
    girth_at_row_with_max(h, row, usize::MAX)
}

pub fn girth_with_max(h: &SparseMatrix, max: usize) -> Option<usize> {
    (0..h.num_cols())
        .filter_map(|c| girth_at_col_with_max(&h, c, max))
        .min()
}

pub fn girth(h: &SparseMatrix) -> Option<usize> {
    girth_with_max(h, usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_girth_circulant() {
        for circulant_size in 2..5 {
            let mut h = SparseMatrix::new(10, 20);
            for j in 0..circulant_size {
                h.insert(j, j);
                h.insert(j, (j + 1) % circulant_size);
            }
            for j in 0..circulant_size {
                assert_eq!(girth_at_col_with_max(&h, j, 10), Some(2 * circulant_size));
                assert_eq!(girth_at_row_with_max(&h, j, 10), Some(2 * circulant_size));
            }
            for j in circulant_size..h.num_cols() {
                assert_eq!(girth_at_col_with_max(&h, j, 10), None);
            }
            for j in circulant_size..h.num_rows() {
                assert_eq!(girth_at_row_with_max(&h, j, 10), None);
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
        assert_eq!(girth_with_max(&h, 100), None);
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
        assert_eq!(girth_with_max(&h, 100), Some(2 * circulant_sizes.1));
        for j in 0..circulant_sizes.0 {
            assert_eq!(
                girth_at_col_with_max(&h, j, 100),
                Some(2 * circulant_sizes.0)
            );
            assert_eq!(
                girth_at_row_with_max(&h, j, 100),
                Some(2 * circulant_sizes.0)
            );
        }
        for j in circulant_sizes.0..(circulant_sizes.0 + circulant_sizes.1) {
            assert_eq!(
                girth_at_col_with_max(&h, j, 100),
                Some(2 * circulant_sizes.1)
            );
            assert_eq!(
                girth_at_row_with_max(&h, j, 100),
                Some(2 * circulant_sizes.1)
            );
        }
        for j in (circulant_sizes.0 + circulant_sizes.1)..h.num_cols() {
            assert_eq!(girth_at_col_with_max(&h, j, 100), None);
        }
        for j in (circulant_sizes.0 + circulant_sizes.1)..h.num_rows() {
            assert_eq!(girth_at_row_with_max(&h, j, 100), None);
        }
    }
}
