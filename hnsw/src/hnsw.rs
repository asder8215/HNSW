use crate::distance::Distance;
use std::{cmp::{min, Ordering}, collections::HashSet, fmt::{Debug, Display}, ptr::NonNull};
use rand::Rng;
use std::collections::BinaryHeap as PriorityQueue;

#[derive(Debug)]
pub struct Node<T> {
    neighbors: PriorityQueue<NonNull<Node<T>>>,
    pub layer: usize,
    value: T
}

impl<T: PartialEq> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
        && self.layer == other.layer
    }
}
impl<T: Eq> Eq for Node<T> { }

impl<T: PartialOrd + Ord> PartialOrd for Node<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Ord> Ord for Node<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.layer.cmp(&other.layer) == Ordering::Equal {
            return self.value.cmp(&other.value)
        }
        return self.layer.cmp(&other.layer)
    }
}

impl<T> Node<T> {
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
    
    /// Gets the neighbors of `self` that are at the same *layer*.
    pub fn neighborhood(&self, layer: usize) -> Box<[NonNull<Self>]> {
        self.neighbors.iter()
            .filter(|n| get_node(n).layer == layer)
            .map(|n| *n)
            .collect()
    }

    pub fn set_neighborhood(&mut self, neighbors: &[NonNull<Self>], layer: usize) {
        // self.neighbors = neighbors
        for neighbor in neighbors.into_iter() {
            let neighbor = get_node_mut(&neighbor);
            if neighbor.layer == layer {
                self.connect_neighbor(neighbor);
            }
        }
    }

    pub fn select_neighbors_simple(&self, c: &[NonNull<Self>], m: usize) -> Box<[NonNull<Self>]> {
        c.iter()
         .take(min(m, c.len()))
         .map(|n| *n)
         .collect()
    }

    pub fn select_neighbors_heuristic(&self,
        c: &[NonNull<Self>],
        m: usize, 
        layer: usize,
        ext_cands: bool,
        keep_pruned_conns: bool
    ) -> Box<[NonNull<Self>]> {
        // TODO: implement algo 4
        self.select_neighbors_simple(c, m)
    }
}

#[derive(Debug)]
pub struct HNSW<T> {
    pub graph: Vec<Node<T>>,
    /// Head contains the Entry Point Node, and the index of the Layer it is at.
    /// tains the Entry Point Node, and the index of the Layer it is at.
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
        let layer_to_insert = (-rand::thread_rng().gen_range(0f64..1f64).ln() * ml).floor() as usize;
        let mut q = Node::new(value, layer_to_insert);
        let mut ep = match self.head {
            Some(head) => head,
            None => {
                self.head = Some(unsafe { NonNull::new_unchecked(&mut q) });
                self.graph.push(q);
                return;
            }
        };
        let top_layer = get_node(&ep).layer;

        for layer in top_layer..(layer_to_insert+1) {
            let nearest_neighbors = Self::search_layer(&q, get_node(&ep), 1, layer);
            // ep = q.get_nearest_from(&nearest_neighbors).expect("nearest_neighbors was empty");
            if let Some(nn) = nearest_neighbors.last() {
                ep = *nn;
            }
        }
        
        for layer in min(top_layer, layer_to_insert)..0 {
            let nearest_neighbors = Self::search_layer(&q, get_node(&ep), ef_construction, layer);
            let neighbors = q.select_neighbors_heuristic(&nearest_neighbors, m, layer, false, false);
            // Add bidirectional connections from neighbors to q
            q.set_neighborhood(&nearest_neighbors, layer);
            
            for neighbor in &neighbors {
                let mut e_conn = get_node(&neighbor).neighborhood(layer);
                if e_conn.len() > mmax {
                    let e_new_conn = get_node(&neighbor)
                        .select_neighbors_heuristic(&mut e_conn, mmax, layer, false, false);
                    for neighbor in e_conn{
                        get_node_mut(&neighbor).set_neighborhood(&e_new_conn, layer);
                    }
                }
            }
            if let Some(nn) = nearest_neighbors.last() {
                ep = *nn;
            }
        }
        if layer_to_insert > top_layer {
            self.head = Some(unsafe { NonNull::new_unchecked(&mut q) });
        }
        self.graph.push(q);
    }

    /// Get a certain number (ef) of neighbors of `ep` that are at `layer`.
    fn search_layer(q: &Node<T>, ep: &Node<T>, ef: usize, layer: usize) -> Box<[NonNull<Node<T>>]> {
        let mut visited_ele = ep.neighbors
            .iter()
            .map(|n| *n)
            .collect::<HashSet<_>>();
        // Reverse PQ because we want to pop the lowest layers first
        let mut candidates = ep.neighbors
            .iter()
            .map(|n| *n)
            .rev()
            .collect::<PriorityQueue<_>>();
        // The elements of w are already sorted because they come from iterating over candidates
        let mut w = candidates
            .iter()
            .map(|n| *n)
            .collect::<Vec<_>>();

        while let Some(c) = candidates.pop() {
            let c = get_node(&c);
            // let f = get_node(&w.pop().expect("Node has no neighbors"));
            let f = w.last().expect("w is empty");
            let f = get_node(&f);

            if c.distance(q) > f.distance(q) {
                break;
            }

            for e in &c.neighbors {
                let e_layer = get_node(e).layer;
                if e_layer == layer {
                    if !visited_ele.contains(&e) {
                        visited_ele.insert(*e);
                        let f = w.last().expect("w is empty");
                        let f = get_node(&f);

                        if get_node(&e).distance(q) < f.distance(q) || w.len() < ef {
                            candidates.push(*e);
                            w.push(*e);
                            if w.len() > ef {
                                w.pop().expect("*prowler-sfx* queue is empty after push?");
                            }
                        }
                    }
                }
            }   
        }

        Box::from(w)
    }
}

impl<T: Debug + Ord> Display for HNSW<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut sorted_graph = self.graph
            .iter()
            .collect::<Vec<_>>();
        sorted_graph.sort();
        f.write_str("HSNW {")?;

        let mut layer = match sorted_graph.first() {
            Some(node) => node.layer,
            None => return f.write_str("}")
        };

        write!(f, "\n    Layer {}: ", layer)?;
        for node in sorted_graph {
            if layer != node.layer {
                layer = node.layer;
                write!(f, "\n    Layer {}: ", layer)?
            }
            write!(f, "{:?}, ", node.value)?;
        }
        
        writeln!(f, "\n}}")
    }
}


pub fn get_node<T>(node: &NonNull<Node<T>>) -> &Node<T> {
    unsafe { & *node.as_ptr() }
}
fn get_node_mut<T>(node: &NonNull<Node<T>>) -> &mut Node<T> {
    unsafe { &mut *node.as_ptr() }
}