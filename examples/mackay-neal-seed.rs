use ldpc_toolbox::mackay_neal;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let m = args[1].parse().unwrap();
    let n = args[2].parse().unwrap();
    let wr = args[3].parse().unwrap();
    let wc = args[4].parse().unwrap();
    let seed = args[5].parse().unwrap();
    let trials = 1000;
    let h = mackay_neal::simple_min_girth(m, n, wr, wc, Some(5), trials, seed).unwrap();
    print!("{}", h.alist());
}
