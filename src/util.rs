use crate::rand::Rng;
use rand::prelude::IteratorRandom;
use std::cmp::Ordering;

pub trait SortedRandomSel {
    type Item;

    fn sort_by_random_sel<F>(self, nitems: usize, compare: F, rng: &mut Rng) -> Option<Self>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering;
}

impl<T> SortedRandomSel for Vec<T> {
    type Item = T;

    fn sort_by_random_sel<F>(mut self, nitems: usize, mut compare: F, rng: &mut Rng) -> Option<Self>
    where
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        if self.len() < nitems {
            return None;
        }
        if nitems == 0 {
            self.clear();
            return Some(self);
        }
        self.sort_unstable_by(|a, b| compare(a, b));
        let cut_element = &self[nitems - 1];
        let sure_cut_idx = self
            .iter()
            .enumerate()
            .take_while(|(_, x)| compare(x, cut_element) == Ordering::Less)
            .last()
            .map_or(0, |x| x.0 + 1);
        let additional_cut_idx = &self[sure_cut_idx..]
            .iter()
            .enumerate()
            .take_while(|(_, x)| compare(x, cut_element) == Ordering::Equal)
            .last()
            .unwrap() // should not panick because this is not an empty iterator
            .0
            + 1;
        let mut sure: Vec<T> = self.drain(0..sure_cut_idx).collect();
        let mut additional = self
            .drain(0..additional_cut_idx)
            .choose_multiple(rng, nitems - sure.len());
        sure.append(&mut additional);
        Some(sure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_sel_all_distinct() {
        let v: Vec<_> = (0..10).collect();
        let u = v
            .sort_by_random_sel(4, |x, y| x.cmp(y), &mut Rng::seed_from_u64(0))
            .unwrap();
        assert_eq!(u, (0..4).collect::<Vec<_>>());
    }

    #[test]
    fn random_sel_equal_by_blocks() {
        let mut v = Vec::new();
        for j in 0..100 {
            v.push((j, j / 10));
        }
        let u = v
            .sort_by_random_sel(15, |(_, x), (_, y)| x.cmp(y), &mut Rng::seed_from_u64(0))
            .unwrap();
        assert_eq!(u.len(), 15);
        let num_zeros = u.iter().filter(|(_, x)| *x == 0).count();
        assert_eq!(num_zeros, 10);
    }

    #[test]
    fn random_sel_all_equal() {
        let mut v = vec![0; 50];
        let u = v
            .sort_by_random_sel(25, |x, y| x.cmp(y), &mut Rng::seed_from_u64(0))
            .unwrap();
        assert_eq!(u.len(), 25);
    }
}
