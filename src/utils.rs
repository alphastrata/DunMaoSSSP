use std::cmp::{Ordering, Reverse};
use std::f64;

#[cfg(not(feature = "hashbrown"))]
use std::collections::BinaryHeap;

#[cfg(feature = "hashbrown")]
use std::collections::BinaryHeap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

pub const INFINITY: f64 = f64::INFINITY;

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct VertexDistance {
    pub vertex: usize,
    pub distance: f64,
}

impl VertexDistance {
    pub fn new(vertex: usize, distance: f64) -> Self {
        VertexDistance { vertex, distance }
    }
}

impl Eq for VertexDistance {}
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for VertexDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.distance.partial_cmp(&other.distance) {
            Some(Ordering::Equal) => Some(self.vertex.cmp(&other.vertex)),
            other => other,
        }
    }
}

impl Ord for VertexDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct AdaptiveDataStructure {
    data: BinaryHeap<Reverse<VertexDistance>>,
    capacity: usize,
    bound: f64,
}

impl AdaptiveDataStructure {
    pub fn new(capacity: usize, bound: f64) -> Self {
        AdaptiveDataStructure {
            data: BinaryHeap::new(),
            capacity,
            bound,
        }
    }

    pub fn insert(&mut self, vertex: usize, distance: f64) {
        if distance < self.bound && distance != INFINITY {
            self.data.push(Reverse(VertexDistance { vertex, distance }));
        }
    }

    pub fn batch_prepend(&mut self, items: Vec<(usize, f64)>) {
        for (vertex, distance) in items {
            self.insert(vertex, distance);
        }
    }

    pub fn pull(&mut self) -> (f64, Vec<usize>) {
        let mut result = Vec::new();
        let mut min_remaining = self.bound;

        while result.len() < self.capacity && !self.data.is_empty() {
            if let Some(Reverse(VertexDistance { vertex, .. })) = self.data.pop() {
                result.push(vertex);

                if let Some(Reverse(VertexDistance {
                    distance: next_dist,
                    ..
                })) = self.data.peek()
                {
                    min_remaining = next_dist.min(min_remaining);
                }
            }
        }

        (min_remaining, result)
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
