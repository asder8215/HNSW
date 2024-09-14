mod hnsw;
mod distance;
use hnsw::Node;
use hnsw::HNSW;
use hnsw::get_node;

fn main() {
    let mut graph = HNSW::new();
    graph.insert(1, 1, 4, 2, 1.5);
    // println!("{:?}", graph);
    // println!("{:?}", *(get_node(&graph.head.unwrap())));
    // println!{"{:?}", get_node(&graph.head.unwrap()).get_neighbors()}
    graph.insert(3, 1, 4, 2, 1.5);
    // println!("{:?}", graph);
    // // println!{"{:?}", get_node(&graph.head.unwrap()).get_neighbors()}
    // graph.insert(2, 1, 4, 2, 1.5);
    // println!("{:?}", graph);
    // // println!{"{:?}", get_node(&graph.head.unwrap()).get_neighbors()}
    // graph.insert(4, 1, 4, 2, 1.5);
    // println!("{:?}", graph);
    // println!{"{:?}", get_node(&graph.head.unwrap()).get_neighbors()}

}
