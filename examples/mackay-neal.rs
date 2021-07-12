use ldpc_toolbox::mackay_neal;

fn main() {
    let h = mackay_neal::simple(6840, 68400, 30, 3, 187).unwrap();
    println!("Alist:");
    println!("{}", h.alist());
    println!("Girth: {:?}", h.girth());
}
