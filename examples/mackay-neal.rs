use ldpc_toolbox::mackay_neal;
use rayon::prelude::*;
use std::fs;

fn main() {
    let m = 844;
    let n = 7595;
    let wr = 30;
    let wc = 3;
    println!("Simple MacKay-Neal with no girth limit");
    let h = (0..1000)
        .into_par_iter()
        .filter_map(|s| mackay_neal::simple(m, n, wr, wc, s).ok())
        .find_any(|_| true)
        .unwrap();
    println!("alist:");
    println!("{}", h.alist());
    println!("Girth: {:?}", h.girth());
    fs::write("/tmp/girth_4.alist", h.alist()).unwrap();

    println!("MacKay-Neal with girth > 4");
    let trials = 1000;
    let h = (0..10000)
        .into_par_iter()
        .filter_map(|s| mackay_neal::simple_min_girth(m, n, wr, wc, Some(5), trials, s).ok())
        .find_any(|_| true)
        .unwrap();
    println!("alist:");
    println!("{}", h.alist());
    println!("Girth: {:?}", h.girth());
    fs::write("/tmp/girth_6.alist", h.alist()).unwrap();
}
