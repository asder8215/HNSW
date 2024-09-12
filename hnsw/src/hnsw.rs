use crate::distance::Distance;
use std::{cmp::min, collections::HashSet, fmt::Debug, ptr::NonNull};
use rand::Rng;
use std::collections::BinaryHeap as PriorityQueue;

#[derive(Debug)]
struct Node<T> {
    neighbors: PriorityQueue<NonNull<Node<T>>>,
    pub layer: usize,
    value: T
}

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
    pub fn neighborhood(&self) -> PriorityQueue<NonNull<Node<T>>> {
        self.neighbors.clone()
    }

    pub fn set_neighborhood(&mut self, neighbors: PriorityQueue<NonNull<Node<T>>>) {
        self.neighbors = neighbors
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
struct HNSW<T> {
    graph: Vec<Node<T>>,
    /// Head contains the Entry Point Node, and the index of the Layer it is at.
    head: Option<NonNull<Node<T>>>
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
        let mut ep = self.head.expect("Can't insert a Node to a Hierarchical Graph with no layers");
        let top_layer = get_node(&ep).layer;
        let layer_to_insert = (-(rand::thread_rng().gen_range(0..1) as f64 * ml).ln()).floor() as usize;
        let mut q = Node::new(value, layer_to_insert);

        for layer in top_layer..(layer_to_insert+1) {
            nearest_neighbors = Self::search_layer(&q, get_node(&ep), 1, layer);
            // ep = q.get_nearest_from(&nearest_neighbors).expect("nearest_neighbors was empty");
            ep = nearest_neighbors.pop().expect("nearest_neighbors was empty");
        }
        
        for layer in min(top_layer, layer_to_insert)..0 {
            nearest_neighbors = Self::search_layer(&q, get_node(&ep), ef_construction, layer);
            let neighbors = q.select_neighbors_heuristic(&mut nearest_neighbors, 
                m, layer, false, false);
            // Add bidirectional connections from neighbors to q
            for neighbor in &neighbors {
                q.connect_neighbor(get_node_mut(neighbor));
            }
            
            for neighbor in &neighbors {
                let mut e_conn = get_node(&neighbor).neighborhood();
                if e_conn.len() > mmax {
                    let e_new_conn = get_node(&neighbor)
                        .select_neighbors_heuristic(&mut e_conn, 
                        mmax, layer, false, false);
                    // todo!() // todo: set neighbor(e) at layer lc to eNewConn
                    get_node_mut(&neighbor).set_neighborhood(e_new_conn);
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
        w.clone()
    }
}


fn get_node<T>(node: &NonNull<Node<T>>) -> &Node<T> {
    unsafe { & *node.as_ptr() }
}
fn get_node_mut<T>(node: &NonNull<Node<T>>) -> &mut Node<T> {
    unsafe { &mut *node.as_ptr() }
}