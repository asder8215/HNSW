mod hnsw;
mod distance;
use hnsw::*;
use rand::Rng;

fn main() {
    let mut graph = HNSW::new();
    // graph.insert(1, 1, 2, 5, 3.0);
    // graph.insert(5, 1, 2, 5, 1.0);
    for _ in 0..6 {
        println!("Inserted Node: {:?}", graph.insert(rand::thread_rng().gen_range(0..10), 3, 1, 3, 1.0));
    }
    let head = graph.head().unwrap();
    println!("Graph: {}", graph);
    println!("Head: {:?}", head);
    // println!("{:?}", graph.head().unwrap().neighbors);
    for node in head.neighbors() {
        println!("Head's neighbor: {:?}", node);
    }
}