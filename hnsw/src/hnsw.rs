use crate::distance::Distance;
use std::{cmp::{min, Ordering}, collections::HashSet, fmt::Debug, ptr::NonNull};
use rand::Rng;
use std::collections::BinaryHeap as PriorityQueue;

// #[derive(Debug, PartialOrd, Ord, Eq)]
#[derive(Debug)]
pub struct Node<T> {
    neighbors: PriorityQueue<NonNull<Node<T>>>,
    pub layer: usize,
    value: T
}

// impl<T> PartialEq for Node<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self == other 
//     }
// }

// impl<T> Eq for Node<T> {}

// impl<T: Ord> PartialOrd for Node<T> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl<T: Ord> Ord for Node<T> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         if self.layer.cmp(&other.layer) == Ordering::Equal {
//             return self.value.cmp(&other.value)
//         }
//         return self.layer.cmp(&other.layer)
//     }
// }

// fn compare<T: Ord>(a: &T, b: &T) -> bool {
//     if a > b{
//         true
//     }else {
//         false
//     }
// }


impl <T: Debug> Node<T> {
    pub fn new(value: T, layer: usize) -> Self {
        Self::new_with_neighbors(value, layer, &[])
    }
    
    pub fn new_with_neighbors(value: T, layer: usize, neighbors: &[NonNull<Self>]) -> Self {
        Self {
            neighbors: PriorityQueue::from(neighbors.to_vec()),
            layer,
            value
        }
    }

    pub fn distance(&self, other: &Self) -> u32
    where T: Distance {
        self.value.distance(&other.value)
    }
    
    pub fn connect_neighbor(&mut self, neighbor: &mut Self) {
        // Connect this node to the neighbor
        self.neighbors.push(unsafe { NonNull::new_unchecked(neighbor) });
        // Connect neighbor to this node
        neighbor.neighbors.push(unsafe { NonNull::new_unchecked(self) });
    }

    // pub fn neighborhood(&self) -> Box<[NonNull<Self>]> {
    //     Box::from_iter(self.neighbors.clone())
    // }
    pub fn get_neighbors(&self) -> PriorityQueue<NonNull<Node<T>>> {
        // let mut neighborhood = PriorityQueue::new();
        // for neighbor in self.neighbors.clone().into_iter() {
        //     if get_node(&neighbor).layer == layer{
        //         neighborhood.push(neighbor);
        //     }
        // }
        self.neighbors.clone()
    }
    
    pub fn neighborhood(&self, layer: usize) -> PriorityQueue<NonNull<Node<T>>> {
        let mut neighborhood = PriorityQueue::new();
        for neighbor in self.neighbors.clone().into_iter() {
            if get_node(&neighbor).layer == layer{
                neighborhood.push(neighbor);
            }
        }
        neighborhood
    }

    pub fn set_neighborhood(&mut self, neighbors: PriorityQueue<NonNull<Node<T>>>, layer: usize) {
        // self.neighbors = neighbors
        for neighbor in neighbors.into_iter() {
            let neighbor = get_node_mut(&neighbor);
            if neighbor.layer == layer {
                self.connect_neighbor(neighbor);
            }
        }
    }

    /// Returns the [`Node`] from the *neighbor list* that has the value closest to `self`.
    /// Will return [`None`] if *neighbors* is empty.
    // pub fn get_nearest_from(&self, neighbors: &[NonNull<Self>]) -> Option<NonNull<Self>> {
    //     let mut nearest = neighbors.first()?;
    //     for neighbor in neighbors {
    //         if distance()
    //     }
    //     todo!()
    // }

    // pub fn select_neighbors_simple(&self, c: &mut PriorityQueue<NonNull<Node<T>>>,
    //          m: usize) -> Box<[NonNull<Self>]> {
    //     let mut nearest_neighbors = PriorityQueue::new();
    //     for _ in 0..min(m, c.len()){
    //         nearest_neighbors.push(c.pop().unwrap())
    //     }
    //     Box::from_iter(nearest_neighbors)
    // }

    pub fn select_neighbors_simple(&self, c: &mut PriorityQueue<NonNull<Node<T>>>,
             m: usize) -> PriorityQueue<NonNull<Node<T>>> {
        let mut nearest_neighbors = PriorityQueue::new();
        for _ in 0..min(m, c.len()){
            nearest_neighbors.push(c.pop().unwrap())
        }
        nearest_neighbors
    }

    // pub fn select_neighbors_heuristic(&self,
    //     c: &mut PriorityQueue<NonNull<Node<T>>>, m: usize, 
    //     layer: usize,
    //     ext_cands: bool, keep_pruned_conns: bool
    // ) -> Box<[NonNull<Self>]> {
    //     // TODO: implement algo 4
    //     self.select_neighbors_simple(c, m)
    // }
    pub fn select_neighbors_heuristic(&self,
        c: &mut PriorityQueue<NonNull<Node<T>>>, m: usize, 
        layer: usize,
        ext_cands: bool, keep_pruned_conns: bool
    ) -> PriorityQueue<NonNull<Node<T>>> {
        // TODO: implement algo 4
        self.select_neighbors_simple(c, m)
    }
}

#[derive(Debug)]
pub struct HNSW<T> {
    pub graph: Vec<Node<T>>,
    /// Head contains the Entry Point Node, and the index of the Layer it is at.
    pub head: Option<NonNull<Node<T>>>
}

impl<T> HNSW<T>
where T: Debug + Distance {
    pub fn new() -> Self {
        Self {
            graph: Vec::new(),
            head: None
        }
    }
    pub fn new_with_graph(graph: Vec<Node<T>>, head: NonNull<Node<T>>) -> Self {
        Self {
            graph,
            head: Some(head)
        }
    }

    pub fn insert(&mut self, value: T, m: usize, mmax: usize, ef_construction: usize, ml:  f64) {
        let mut nearest_neighbors: PriorityQueue<NonNull<Node<T>>>;
        let head = self.head;
        let layer_to_insert = (-(rand::thread_rng().gen_range(0f64..1f64).ln()) * ml).floor() as usize;
        let mut q = Node::new(value, layer_to_insert);
        println!("Query inserting: {:?}", q);
        let mut ep;
        if head == None {
            self.head = Some(unsafe { NonNull::new_unchecked(&mut q) });
            // self.graph.push(q);
            return;
        }
        
        ep = head.unwrap();
        let top_layer = get_node(&ep).layer;

        // println!("{:?}", top_layer);
        // return;

        for layer in top_layer..(layer_to_insert+1) {
            nearest_neighbors = Self::search_layer(&q, get_node(&ep), 1, layer);
            // ep = q.get_nearest_from(&nearest_neighbors).expect("nearest_neighbors was empty");
            if nearest_neighbors.len() != 0 {
                ep = nearest_neighbors.pop().expect("nearest_neighbors was empty");
            }
        }
        
        for layer in min(top_layer, layer_to_insert)..0 {
            nearest_neighbors = Self::search_layer(&q, get_node(&ep), ef_construction, layer);
            let neighbors = q.select_neighbors_heuristic(&mut nearest_neighbors, 
                m, layer, false, false);
            // Add bidirectional connections from neighbors to q
            for neighbor in &neighbors {
                let neighbor = get_node_mut(neighbor);
                if neighbor.layer == layer {
                    q.connect_neighbor(neighbor);
                }
            }
            
            for neighbor in &neighbors {
                let mut e_conn = get_node(&neighbor).neighborhood(layer);
                if e_conn.len() > mmax {
                    let e_new_conn = get_node(&neighbor)
                        .select_neighbors_heuristic(&mut e_conn, 
                        mmax, layer, false, false);
                    // todo!() // todo: set neighbor(e) at layer lc to eNewConn
                    for neighbor in e_conn{
                        get_node_mut(&neighbor).set_neighborhood(e_new_conn.clone(), layer);
                    }
                }
            }
            ep = nearest_neighbors.pop().expect("nearest_neighbors was empty");
        }
        if layer_to_insert > top_layer {
            self.head = Some(unsafe { NonNull::new_unchecked(&mut q) });
        }
    }

    fn search_layer(q: &Node<T>, ep: &Node<T>, ef: usize, layer: usize) -> PriorityQueue<NonNull<Node<T>>> {
        let visited_ele = &mut ep.neighbors.clone()
            .into_iter()
            .collect::<HashSet<_>>();
        let candidates = &mut ep.neighbors.iter()
            .rev()
            .collect::<PriorityQueue<_>>();
        let w = &mut ep.neighbors.clone();

        while candidates.len() > 0 {
            let c = get_node(&candidates.pop().expect("Node has no neighbors"));
            // let f = get_node(&w.pop().expect("Node has no neighbors"));
            let f = &w.pop().expect("Node has no neighbors");

            if c.distance(q) > get_node(f).distance(q) {
                break;
            }

            for e in &c.neighbors {
                let e_layer = get_node(e).layer;
                if e_layer == layer{
                    if !visited_ele.contains(&e){
                        visited_ele.insert(*e);
                        let f = &w.pop().expect("Node has no neighbors");

                        if get_node(&e).distance(q) < get_node(f).distance(q) || w.len() < ef {
                            candidates.push(&e);
                            w.push(*e);
                            if w.len() > ef {
                                w.pop().expect("Node has no neighbors");
                            }
                        }
                    }
                }
            } 
        }
        w.clone()
    }
}


pub fn get_node<T>(node: &NonNull<Node<T>>) -> &Node<T> {
    unsafe { & *node.as_ptr() }
}
fn get_node_mut<T>(node: &NonNull<Node<T>>) -> &mut Node<T> {
    unsafe { &mut *node.as_ptr() }
}