use crate::distance::Distance;
use std::{cmp::{max, min, Ordering}, collections::HashSet, fmt::{Debug, Display}, ops::DerefMut, pin::Pin, ptr::NonNull};
use rand::Rng;
use std::collections::BinaryHeap as PriorityQueue;
use std::hash::Hash;

/// Reference without the lifetime.
/// Use this instead of NonNull so that neighbors can be sorted in the priority queue.
#[derive(Debug)]
struct Ref<T>(NonNull<T>);
impl<T> Ref<T> {
    pub fn new(value: &T) -> Self {
        Self(unsafe { NonNull::new_unchecked(value as *const _ as *mut _) })
    }

    pub fn get_val(&self) -> &T {
        unsafe { & *self.0.as_ptr() }
    }

    pub fn get_val_mut(&self) -> &mut T {
        unsafe { &mut *self.0.as_ptr() }
    }
}

impl<T> Clone for Ref<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> Copy for Ref<T> {}

impl<T: PartialEq> PartialEq for Ref<T> {
    fn eq(&self, other: &Self) -> bool {
        let this = self.get_val();
        let other = other.get_val();
        this.eq(other)
    }
}
impl<T: Eq> Eq for Ref<T> { }

impl<T: PartialOrd + Ord> PartialOrd for Ref<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Ord> Ord for Ref<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        let this = self.get_val();
        let other = other.get_val();
        this.cmp(other)
    }
}

impl<T> Hash for Ref<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}


#[derive(Debug)]
pub struct Node<T: Ord> {
    neighbors: PriorityQueue<Ref<Node<T>>>,
    layer: usize,
    value: T
}

impl<T: PartialEq + Ord> PartialEq for Node<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
        && self.layer == other.layer
    }
}
impl<T: Eq + Ord> Eq for Node<T> { }

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

impl<T: Ord> Node<T> {
    pub fn new(value: T, layer: usize) -> Self {
        // Self::new_with_neighbors(value, layer, &[])
        Self {
            neighbors: PriorityQueue::new(),
            layer,
            value
        }
    }
    // pub fn new_with_neighbors(value: T, layer: usize, neighbors: &[NonNull<Self>]) -> Self {
    //     Self {
    //         neighbors: PriorityQueue::from(neighbors.to_vec()),
    //         layer,
    //         value
    //     }
    // }

    pub fn neighbors(&self) -> Box<[&Self]> {
        self.neighbors.iter()
            .map(|n| n.get_val())
            .collect()
    }

    pub fn distance(&self, other: &Self) -> u32
    where T: Distance {
        self.value.distance(&other.value)
    }
    
    pub fn connect_neighbor(self: Pin<&mut Self>, neighbor: Pin<&mut Self>) {
        let self_ref = Ref::new(self.as_ref().get_ref());
        let other_ref = Ref::new(neighbor.as_ref().get_ref());
        // Connect this node to the neighbor
        unsafe { self.get_unchecked_mut() }.neighbors.push(other_ref);
        // Connect neighbor to this node
        unsafe { neighbor.get_unchecked_mut() }.neighbors.push(self_ref);
    }
    
    /// Gets the neighbors of `self` that are at the same *layer*.
    fn neighborhood(&self, layer: usize) -> Box<[Ref<Self>]> {
        self.neighbors.iter()
            .filter(|n| n.get_val().layer == layer)
            .map(|n| *n)
            .collect()
    }

    fn set_neighborhood(self: Pin<&mut Self>, neighbors: &[Ref<Self>], layer: usize) {
        let this = unsafe { self.get_unchecked_mut() };
        for neighbor in neighbors.into_iter() {
            let neighbor = unsafe { Pin::new_unchecked(neighbor.get_val_mut()) };
            if neighbor.layer == layer {
                unsafe { Pin::new_unchecked(&mut *this) }.connect_neighbor(neighbor);
            }
        }
    }

    fn select_neighbors_simple(&self, c: &[Ref<Self>], m: usize) -> Box<[Ref<Self>]> {
        c.iter()
         .take(min(m, c.len()))
         .map(|n| *n)
         .collect()
    }

    fn select_neighbors_heuristic(&self,
        c: &[Ref<Self>],
        m: usize, 
        layer: usize,
        ext_cands: bool,
        keep_pruned_conns: bool
    ) -> Box<[Ref<Self>]> {
        // TODO: implement algo 4
        self.select_neighbors_simple(c, m)
    }
}

#[derive(Debug)]
pub struct HNSW<T: Ord> {
    graph: Vec<Pin<Box<Node<T>>>>,
    /// Head contains the Entry Point Node, and the index of the Layer it is at.
    /// tains the Entry Point Node, and the index of the Layer it is at.
    head: Option<Ref<Node<T>>>
}

impl<T: Ord> HNSW<T>
where T: Debug + Distance {
    pub fn new() -> Self {
        Self {
            graph: Vec::new(),
            head: None
        }
    }
    // pub fn new_with_graph(graph: Vec<Node<T>>, head: NonNull<Node<T>>) -> Self {
    //     Self {
    //         graph,
    //         head: Some(head)
    //     }
    // }

    /// Insert a new element into hsnw
    /// 
    /// `q`: element to insert
    /// 
    /// `m`: number of established connections, maximum number of connections for each element
    /// 
    /// `mmax`: maximum number of connections for each element per layer 
    /// 
    /// `ef_construction`: size of the dynamic candidate list
    /// 
    /// `ml`: normalization factor for level generation
    pub fn insert(&mut self, value: T, m: usize, mmax: usize, ef_construction: usize, ml:  f64) -> &Node<T> {
        let mut nearest_neighbors;
        // generating element's new level
        let layer_to_insert = (-rand::thread_rng().gen_range(0f64..1f64).ln() * ml).floor() as usize;
        // insert it into the graph (for debugging purposes)
        self.graph.push(Box::pin(Node::new(value, layer_to_insert)));
        // grab the same reference of the last element in graph
        let q = self.graph.last_mut().unwrap();

        // if head is empty (nothing was in the graph), then create the head
        // otherwise, head is the entry point 
        let mut ep = match self.head {
            Some(head) => head,
            None => {
                self.head = Some(Ref::new(q));
                return q;
            }
        };
        let top_layer = ep.get_val().layer;

        for layer in min(top_layer, layer_to_insert+1)..max(top_layer, layer_to_insert + 1) {
            let nearest_neighbors = Self::search_layer(&q, ep.get_val_mut(), 1, layer);
            if let Some(nn) = nearest_neighbors.last() {
                ep = *nn;
            }
        }

        q.as_mut().connect_neighbor(unsafe { Pin::new_unchecked(ep.get_val_mut()) });

        for layer in (0..min(top_layer, layer_to_insert)).rev() {
            nearest_neighbors = Self::search_layer(&q, ep.get_val_mut(), ef_construction, layer);
            let neighbors = q.select_neighbors_heuristic(&nearest_neighbors, m, layer, false, false);
            // Add bidirectional connections from neighbors to q
            q.as_mut().set_neighborhood(&neighbors, layer);

            for neighbor in &neighbors {
                let e = neighbor.get_val_mut();
                let mut e_conn = e.neighborhood(layer);
                if e_conn.len() > mmax {
                    let e_new_conn = e
                        .select_neighbors_heuristic(&mut e_conn, mmax, layer,
                             false, false);
                    
                    e.neighbors = e_new_conn
                        .iter()
                        .map(|n| *n)
                        .collect::<PriorityQueue<_>>();
                }
            }
            if let Some(nn) = nearest_neighbors.last() {
                ep = *nn;
            }
        }
        if layer_to_insert > top_layer {
            self.head = Some(Ref::new(q));
        }
        q
    }

    /// Get a certain number (ef) of neighbors of `ep` that are at `layer`.
    fn search_layer(q: &Node<T>, ep: &mut Node<T>, ef: usize, layer: usize) -> Box<[Ref<Node<T>>]> {
        let mut visited_ele = HashSet::<Ref<Node<T>>>::new();
        let mut candidates = PriorityQueue::<Ref<Node<T>>>::new();
        let mut w = Vec::<Ref<Node<T>>>::new();

        visited_ele.insert(Ref::new(&*ep));
        candidates.push(Ref::new(&*ep));
        w.push(Ref::new(&*ep));

        while let Some(c) = candidates.pop() {
            let c = c.get_val();
            let f = w.last().expect("w is empty").get_val();

            if c.distance(q) > f.distance(q) {
                break;
            }

            for e in c.neighborhood(layer) {
                if !visited_ele.contains(&e) {
                    visited_ele.insert(e);
                    let f = w.last().expect("W is empty").get_val();

                    if e.get_val().distance(q) < f.distance(q) || w.len() < ef {
                        candidates.push(e);
                        w.push(e);
                        if w.len() > ef {
                            w.pop().expect("Queue is empty after push");
                        }
                    }
                }
            }   
        }

        Box::from(w)
    }

    pub fn head(&self) -> Option<&Node<T>> {
        match &self.head {
            Some(head) => Some(head.get_val()),
            None => None
        }
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
