use std::{cmp::min, fmt::Debug, ptr::NonNull};
use rand::Rng;
use std::collections::BinaryHeap as PriorityQueue;

#[derive(Debug)]
struct Node<T> {
    // neighbors: PriorityQueue<Edge<T>>,
    neighbors: PriorityQueue<NonNull<Node<T>>>,
    layer_up: Option<NonNull<Node<T>>>,
    layer_down: Option<NonNull<Node<T>>>,
    value: T,
    vector: Vec<f64>
}

// #[derive(Debug)]
// struct Edge<T> {
//     neighbor: NonNull<Node<T>>,
//     weight: u64
// }
// impl <T> Clone for Edge<T> {
//     fn clone(&self) -> Self {
//         Self {
//             neighbor: self.neighbor,
//             weight: self.weight
//         }
//     }
// }
// impl <T> Copy for Edge<T> { }
// impl <T> PartialEq for Edge<T> {
//     fn eq(&self, other: &Self) -> bool {
//         self.neighbor == other.neighbor
//         && self.weight == other.weight
//     }
// }
// impl <T> Eq for Edge<T> { }
// impl <T> PartialOrd for Edge<T> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }
// impl <T> Ord for Edge<T> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.weight.cmp(&other.weight)
//     }
// }


#[derive(Debug)]
pub struct Graph<T> {
    nodes: Vec<Node<T>>
}

#[derive(Debug)]
struct HNSW<T> {
    layers: Vec<Graph<T>>,
    /// Head contains the Entry Point Node, and the index of the Layer it is at.
    head: Option<(NonNull<Node<T>>, usize)>
}

impl <T: Debug> Node<T> {
    pub fn new(value: T) -> Self {
        Self::new_with_neighbors(value, &[])
    }
    
    pub fn new_with_neighbors(value: T, neighbors: &[Edge<T>]) -> Self {
        Self {
            neighbors: PriorityQueue::from(neighbors.to_vec()),
            layer_up: Some(unsafe { NonNull::new_unchecked(std::ptr::null_mut())}),
            layer_down: Some(unsafe { NonNull::new_unchecked(std::ptr::null_mut())}),
            value: value
        }
    }
    
    /// Connects 2 Nodes that are in the same layer.
    ///
    /// Don't use this to connect nodes that are in different layers.
    /// Use [`Self::connect_to_bottom_layer()`] instead.
    pub fn connect_neighbor(&mut self, neighbor: &mut Self, weight: Option<u64>) {
        let weight = weight.unwrap_or(0);
        // Connect this node to the neighbor
        self.neighbors.push(Edge {
            neighbor: unsafe { NonNull::new_unchecked(neighbor) },
            weight
        });
        // Connect neighbor to this node
        neighbor.neighbors.push(Edge {
            neighbor: unsafe { NonNull::new_unchecked(self) },
            weight
        });
    }

    /// Connects Nodes that are in different layers,
    /// where `self` is a Node on a layer higher than `other`.
    ///
    /// Will `panic!` if the layer_down or layer_up is not null.
    pub fn connect_to_bottom_layer(&mut self, other: &mut Self) {
        if let Some(node) = self.layer_down {
            panic!("Self is already connected to a Node in a lower layer: {:?}", unsafe { node.as_ref() })
        }
        if let Some(node) = other.layer_up {
            panic!("The other Node is already connected to a Node in a higher layer: {:?}", unsafe { node.as_ref() })
        }

        // Connect this node to the neighbor
        self.layer_down = unsafe { Some(NonNull::new_unchecked(other)) };
        // Connect neighbor to this node
        other.layer_up = unsafe { Some(NonNull::new_unchecked(self)) };
    }

    pub fn select_neighbors_simple(&self, c: &[NonNull<Self>], m: u64) -> Box<[Edge<T>]> {
        self.neighbors
            .iter()
            .map(|edge| *edge)
            .filter(|edge| c.contains(&edge.neighbor))
            .take(m as usize)
            .collect()
    }

    pub fn select_neighbors_heuristic(&self,
        c: &[NonNull<Self>], m: u64, 
        layer: usize,
        ext_cands: bool, keep_pruned_conns: bool
    ) -> Box<[Edge<T>]> {
        // TODO: implement algo 4
        self.select_neighbors_simple(c, m)
    }
}

impl<T:Debug> HNSW<T> {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            head: None
        }
    }
    pub fn new_with_layers(layers: Vec<Graph<T>>, head: NonNull<Node<T>>, head_layer: usize) -> Self {
        Self {
            layers,
            head: Some((head, head_layer))
        }
    }

    pub fn insert(&self, q: Node<T>, m: u64, mmax: u64, ef_construction: u64, ml:  f64) {
        let mut w: Vec<Node<T>>;
        let head = self.head.expect("Can't insert a Node to a Hierarchical Graph with no layers");
        let ep = head.0;
        let top_layer = head.1;
        let low_layer = (-(rand::thread_rng().gen_range(0..1) as f64 * ml).ln()).floor() as usize;

        for layer in top_layer..(low_layer+1) {
            w = Self::search_layer(q, ep, 1, layer);
            ep = Self::get_nearest_neighbor()
        }
        
        for layer in min(top_layer, low_layer)..0 {
            w = Self::search_layer(&q, ep, ef_construction, layer);
            let neighbors = q.select_neighbors_heuristic(w, m, layer, false, false);
            // Add bidirectional connections from neighbors to q
            for edge in neighbors {
                q.connect_neighbor(get_node_mut(&edge.neighbor), None);
            }
            
            for neighbor in neighbors {
                let e_conn = Self::neighborhood(neighbor);
                if e_conn.len() > mmax {
                    e_new_conn = Self::select_neighbors_heuristic(neighbor, e_conn, mmax, layer, false, false)
                    // todo: set neighbor(e) at layer lc to eNewConn
                }
            }
            ep = w.first();
        }
    }

    fn search_layer(q: &Node<T>, ep: NonNull<Node<T>>, ef: u64, layer: usize) -> Vec<Node<T>> {
        let ep = get_node(&ep);
        let visited_ele = &mut ep.neighbors.clone()
            .into_iter()
            .map(|edge| edge.neighbor)
            .collect::<Vec<_>>();
        let candidates = &mut ep.neighbors.clone();
        let w = &mut ep.neighbors.clone();

        while candidates.len() > 0 {
            let c = *candidates.iter().last().expect("Node has no neighbors");
            candidates.retain(|edge| edge.neighbor != c.neighbor);
            let f = w.pop().expect("Node has no neighbors");

            if c.weight > f.weight {
                break;
            }

            for e in neighbourhood(c.neighbor) {
                if visited_ele.contains(e)
            }
            
        }

        todo!()
    }
    fn get_nearest_neighbor() -> NonNull<Node<T>> {
        todo!()
    }
    fn neighborhood() -> u64 {
        todo!()
    }
}

fn get_node<T>(node: &NonNull<Node<T>>) -> &Node<T> {
    unsafe { & *node.as_ptr() }
}
fn get_node_mut<T>(node: &NonNull<Node<T>>) -> &mut Node<T> {
    unsafe { &mut *node.as_ptr() }
}