use hnsw::hnsw::HNSW;
use rand::Rng;


#[test]
fn print_graph() {
    println!("\nTesting Graph");
    let mut graph = HNSW::new();
    graph.insert(1, 5, 4, 2, 1.0);
    graph.insert(5, 5, 4, 2, 1.0);
    graph.insert(3, 5, 4, 2, 1.0);
    graph.insert(2, 5, 4, 2, 1.0);
    println!("{}", graph);
}

#[test]
fn nodes_connection() {
    println!("\nTesting if nodes in graph are connected");
    let mut graph = HNSW::new();
    println!("{:?}", graph.insert(2, 5, 4, 2, 1.0));
    graph.insert(5, 5, 4, 2, 1.0);
    graph.insert(3, 5, 4, 2, 1.0);
    graph.insert(4, 5, 4, 2, 1.0);

    // for _ in 0..6 {
    //     println!("{:?}", graph.insert(rand::thread_rng().gen_range(0..1), 1, 2, 5, 1.0));
    // }
    // let head = graph.head().unwrap();
    println!("{}", graph);
    // println!("{:?}", head);
    // for node in graph.graph {
    //     println!("{:?}", node)
    // }
    // println!("{:?}", head.neighbors);
    // let head = graph.head;
    panic!("stf")
}
