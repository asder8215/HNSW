use hnsw::hnsw::HNSW;


#[test]
fn feature() {
    println!("Testing Graph");
    let mut graph = HNSW::new();
    graph.insert(1, 5, 4, 2, 1.0);
    graph.insert(5, 5, 4, 2, 1.0);
    graph.insert(3, 5, 4, 2, 1.0);
    graph.insert(2, 5, 4, 2, 1.0);
    println!("{}", graph.graph.len());
    println!("{}", graph);
    panic!("TESTING")
}
