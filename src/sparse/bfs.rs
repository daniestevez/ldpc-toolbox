use crate::sparse::{Node, SparseMatrix};
use std::collections::VecDeque;

#[derive(Debug, Clone, Eq, PartialEq)]
struct PathHead {
    node: Node,
    parent: Option<Node>,
    path_length: usize,
}

impl PathHead {
    fn iter<'a>(&'a self, h: &'a SparseMatrix) -> impl Iterator<Item = PathHead> + 'a {
        self.node
            .iter(h)
            .filter(move |&x| {
                if let Some(parent) = self.parent {
                    x != parent
                } else {
                    true
                }
            })
            .map(move |x| PathHead {
                node: x,
                parent: Some(self.node),
                path_length: self.path_length + 1,
            })
    }
}

/// Results for BFS algorithm
///
/// This gives the distances of each of the nodes of the graph from the node
/// that was used as root for the BFS algorithm. Distances are represented
/// as `Option<usize>`, with the value `None` for nodes that are not reachable
/// from the root.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BFSResults {
    /// The vector of distances from each of the row nodes to the root
    pub row_nodes_distance: Vec<Option<usize>>,
    /// The vector of distances from each of the column nodes to the root
    pub col_nodes_distance: Vec<Option<usize>>,
}

impl BFSResults {
    fn get_node_mut(&mut self, node: Node) -> &mut Option<usize> {
        match node {
            Node::Row(n) => &mut self.row_nodes_distance[n],
            Node::Col(n) => &mut self.col_nodes_distance[n],
        }
    }
}

pub struct BFSContext<'a> {
    results: BFSResults,
    to_visit: VecDeque<PathHead>,
    h: &'a SparseMatrix,
}

impl<'a> BFSContext<'a> {
    pub fn new(h: &'a SparseMatrix, node: Node) -> Self {
        let mut to_visit = VecDeque::new();
        to_visit.push_back(PathHead {
            node,
            parent: None,
            path_length: 0,
        });
        let mut results = BFSResults {
            row_nodes_distance: vec![None; h.num_rows()],
            col_nodes_distance: vec![None; h.num_cols()],
        };
        results.get_node_mut(node).replace(0);
        BFSContext {
            results,
            to_visit,
            h,
        }
    }

    pub fn bfs(mut self) -> BFSResults {
        while let Some(head) = self.to_visit.pop_front() {
            for next_head in head.iter(self.h) {
                let next_dist = self.results.get_node_mut(next_head.node);
                if next_dist.is_none() {
                    *next_dist = Some(next_head.path_length);
                    self.to_visit.push_back(next_head);
                }
            }
        }
        self.results
    }

    pub fn local_girth(mut self, max: usize) -> Option<usize> {
        while let Some(head) = self.to_visit.pop_front() {
            for next_head in head.iter(self.h) {
                let next_dist = self.results.get_node_mut(next_head.node);
                if let Some(dist) = *next_dist {
                    let total = dist + next_head.path_length;
                    return if total <= max { Some(total) } else { None };
                } else {
                    *next_dist = Some(next_head.path_length);
                    if next_head.path_length < max {
                        self.to_visit.push_back(next_head);
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disconnected_2x2() {
        let mut h = SparseMatrix::new(2, 2);
        h.insert(0, 0);
        h.insert(1, 1);
        let r = h.bfs(Node::Col(0));
        assert_eq!(r.row_nodes_distance[0], Some(1));
        assert_eq!(r.row_nodes_distance[1], None);
        assert_eq!(r.col_nodes_distance[0], Some(0));
        assert_eq!(r.col_nodes_distance[1], None);
    }

    #[test]
    fn complete_nxm() {
        let n = 20;
        let m = 10;
        let mut h = SparseMatrix::new(n, m);
        for i in 0..n {
            for j in 0..m {
                h.insert(i, j);
            }
        }
        let r = h.bfs(Node::Row(0));
        assert_eq!(r.row_nodes_distance[0], Some(0));
        for i in 1..n {
            assert_eq!(r.row_nodes_distance[i], Some(2));
        }
        for i in 0..m {
            assert_eq!(r.col_nodes_distance[i], Some(1));
        }
    }

    #[test]
    fn circulant() {
        let n = 20;
        let mut h = SparseMatrix::new(n, n);
        for j in 0..n {
            h.insert(j, j);
            h.insert(j, (j + 1) % n);
        }
        let r = h.bfs(Node::Row(0));
        assert_eq!(r.row_nodes_distance[0], Some(0));
        for j in 1..n {
            let dist = std::cmp::min(2 * j, 2 * (n - j));
            assert_eq!(r.row_nodes_distance[j], Some(dist));
        }
        for j in 1..n + 1 {
            let dist = std::cmp::min(2 * j - 1, 2 * (n - j) + 1);
            assert_eq!(r.col_nodes_distance[j % n], Some(dist));
        }
    }
}
