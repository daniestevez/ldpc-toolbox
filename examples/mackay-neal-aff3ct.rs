use ldpc_toolbox::mackay_neal;
use std::fs;
use std::process::Command;

const ALIST_PATH: &str = "/tmp/alist";

fn run_aff3ct(meta: &str) -> Vec<u8> {
    Command::new("aff3ct")
        .args(&[
            "--sim-type",
            "BFER",
            "-C",
            "LDPC",
            "--enc-type",
            "LDPC_H",
            "-m",
            "9.4",
            "-M",
            "9.4",
            "--dec-implem",
            "SPA",
            "--dec-h-path",
            ALIST_PATH,
            "-e",
            "100",
            "-i",
            "2000",
            "--dec-type",
            "BP_FLOODING",
            "--mdm-const-path",
            "32apsk.mod",
            "--mdm-type",
            "USER",
            "--mdm-max",
            "MAXSS",
            "--sim-meta",
            meta,
        ])
        .output()
        .expect("failed to run aff3ct")
        .stdout
}

fn process_seed(seed: u64) {
    let m = 844;
    let n = 7595;
    let wr = 27;
    let wc = 3;
    let trials = 1000;

    if let Ok(h) = mackay_neal::simple_min_girth(m, n, wr, wc, Some(5), trials, seed) {
        println!("seed = {}. MacKay-Neal successful. Running aff3ct...", seed);
        fs::write(ALIST_PATH, h.alist()).expect("unable to write alist file");
        let metadata = format!("seed={}", seed);
        let out = run_aff3ct(&metadata);
        let filename = format!("{}.txt", seed);
        fs::write(filename, out).expect("unable to write aff3ct log");
    } else {
        println!("seed = {}. MacKay-Neal unsuccessful", seed);
    }
}

fn main() {
    for s in 0.. {
        process_seed(s);
    }
}
