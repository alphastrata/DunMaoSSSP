use std::cmp::{Ordering, Reverse};
use std::f64;

#[cfg(not(feature = "hashbrown"))]
use std::collections::{BinaryHeap, HashMap, HashSet};

#[cfg(feature = "hashbrown")]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "hashbrown")]
use std::collections::BinaryHeap;

use bit_vec::BitVec;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Graph {
    pub vertices: usize,
    pub edges: Vec<Vec<Edge>>,
}

impl Graph {
    pub fn new(vertices: usize) -> Self {
        Graph {
            vertices,
            edges: vec![Vec::new(); vertices],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges[from].push(Edge { to, weight });
    }

    pub fn edge_count(&self) -> usize {
        self.edges.iter().map(|adj| adj.len()).sum()
    }

    #[cfg(feature = "bincode")]
    pub fn from_file(path: &std::path::Path) -> Result<Graph, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let config = bincode::config::legacy();
        let reader = std::io::BufReader::new(file);
        let graph = bincode::decode_from_reader(reader, config)?;
        Ok(graph)
    }

    #[cfg(feature = "bincode")]
    pub fn to_file(
        graph: &Graph,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let config = bincode::config::legacy();
        let mut writer = std::io::BufWriter::new(file);

        bincode::serde::encode_into_std_write(graph, &mut writer, config)?;

        Ok(())
    }
}

// Public API functions inspired by petgraph

/// Computes the shortest path from a source to a goal using the Dun-Mao algorithm.
pub fn dun_mao_shortest_path(
    graph: &Graph,
    source: usize,
    goal: usize,
) -> Option<(f64, Vec<usize>)> {
    let mut solver = SSSpSolver::new(graph.clone());
    solver.solve(source, goal)
}

/// Computes all distances from a source vertex using the Dun-Mao algorithm.
pub fn dun_mao_all_distances(graph: &Graph, source: usize) -> HashMap<usize, f64> {
    let mut solver = SSSpSolver::new(graph.clone());
    solver.solve_all(source)
}

#[derive(Debug, Clone, PartialEq)]
struct VertexDistance {
    vertex: usize,
    distance: f64,
}

impl Eq for VertexDistance {}

impl Ord for VertexDistance {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl PartialOrd for VertexDistance {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Implements the SSSP algorithm from "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
/// by Duan, Mao, Mao, Shu, and Yin (2025).
pub struct SSSpSolver {
    graph: Graph,
    /// Stores the shortest distance from the source to each vertex.
    distances: Vec<f64>,
    /// Stores the predecessor of each vertex in the shortest path.
    predecessors: Vec<Option<usize>>,
    /// A bitmask to mark vertices as complete (visited and finalized).
    complete: BitVec,
    /// Parameter `k`, approximately log^(1/3)(n).
    k: usize,
    /// Parameter `t`, approximately log^(2/3)(n).
    t: usize,
}

impl SSSpSolver {
    pub fn new(graph: Graph) -> Self {
        let n = graph.vertices;
        let k = ((n as f64).ln().powf(1.0 / 3.0)).floor() as usize;
        let t = ((n as f64).ln().powf(2.0 / 3.0)).floor() as usize;

        SSSpSolver {
            distances: vec![f64::INFINITY; n],
            predecessors: vec![None; n],
            complete: BitVec::from_elem(n, false),
            graph,
            k: k.max(2), // Ensure k is at least 2 for small graphs
            t: t.max(2), // Ensure t is at least 2 for small graphs
        }
    }

    pub fn solve(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        self.distances[source] = 0.0;
        self.complete.set(source, true);

        // For small graphs, use simple Dijkstra
        if self.graph.vertices <= 10 {
            return self.dijkstra(source, Some(goal));
        }

        let max_level = ((self.graph.vertices as f64).ln() / self.t as f64).ceil() as usize;
        let frontier = vec![source];

        let (_, _result) = self.bmssp(max_level, f64::INFINITY, frontier);

        // After bmssp, some nodes might be reachable but not yet finalized.
        // Run a Dijkstra-like process on the entire graph to settle all nodes.
        // This uses the distances computed by bmssp as a starting point.
        let mut heap = BinaryHeap::new();
        for i in 0..self.graph.vertices {
            if !self.complete.get(i).unwrap_or(false) && self.distances[i] != f64::INFINITY {
                heap.push(Reverse(VertexDistance {
                    vertex: i,
                    distance: self.distances[i],
                }));
            }
        }

        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            if u == goal {
                break; // Goal reached
            }

            if dist > self.distances[u] {
                continue;
            }

            self.complete.set(u, true);

            self.graph.edges[u].iter().for_each(|edge| {
                let v = edge.to;
                let new_dist = dist + edge.weight;

                if new_dist < self.distances[v] {
                    self.distances[v] = new_dist;
                    self.predecessors[v] = Some(u);
                    heap.push(Reverse(VertexDistance {
                        vertex: v,
                        distance: new_dist,
                    }));
                }
            });
        }

        if self.distances[goal] == f64::INFINITY {
            None
        } else {
            Some((self.distances[goal], self.reconstruct_path(source, goal)))
        }
    }

    pub fn solve_all(&mut self, source: usize) -> HashMap<usize, f64> {
        self.distances[source] = 0.0;
        self.complete.set(source, true);

        if self.graph.vertices <= 10 {
            self.dijkstra(source, None);
        } else {
            let max_level = ((self.graph.vertices as f64).ln() / self.t as f64).ceil() as usize;
            let frontier = vec![source];
            self.bmssp(max_level, f64::INFINITY, frontier);

            let mut heap = BinaryHeap::new();
            for i in 0..self.graph.vertices {
                if !self.complete.get(i).unwrap_or(false) && self.distances[i] != f64::INFINITY {
                    heap.push(Reverse(VertexDistance {
                        vertex: i,
                        distance: self.distances[i],
                    }));
                }
            }

            while let Some(Reverse(VertexDistance {
                vertex: u,
                distance: dist,
            })) = heap.pop()
            {
                if dist > self.distances[u] {
                    continue;
                }
                self.complete.set(u, true);
                for edge in &self.graph.edges[u] {
                    let v = edge.to;
                    let new_dist = dist + edge.weight;
                    if new_dist < self.distances[v] {
                        self.distances[v] = new_dist;
                        self.predecessors[v] = Some(u);
                        heap.push(Reverse(VertexDistance {
                            vertex: v,
                            distance: new_dist,
                        }));
                    }
                }
            }
        }

        let mut results = HashMap::new();
        for i in 0..self.graph.vertices {
            if self.distances[i] != f64::INFINITY {
                results.insert(i, self.distances[i]);
            }
        }
        results
    }

    fn reconstruct_path(&self, source: usize, goal: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = goal;
        while current != source {
            path.push(current);
            if let Some(pred) = self.predecessors[current] {
                current = pred;
            } else {
                // Should not happen if a path exists
                return Vec::new();
            }
        }
        path.push(source);
        path.reverse();
        path
    }

    pub fn dijkstra(&mut self, source: usize, goal: Option<usize>) -> Option<(f64, Vec<usize>)> {
        let mut heap = BinaryHeap::new();
        self.distances[source] = 0.0;
        heap.push(Reverse(VertexDistance {
            vertex: source,
            distance: 0.0,
        }));

        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            if let Some(g) = goal
                && u == g
            {
                break; // Goal reached
            }

            if dist > self.distances[u] {
                continue;
            }

            self.complete.set(u, true);

            self.graph.edges[u].iter().for_each(|edge| {
                let v = edge.to;
                let new_dist = dist + edge.weight;

                if new_dist < self.distances[v] {
                    self.distances[v] = new_dist;
                    self.predecessors[v] = Some(u);
                    heap.push(Reverse(VertexDistance {
                        vertex: v,
                        distance: new_dist,
                    }));
                }
            });
        }

        if let Some(g) = goal {
            if self.distances[g] == f64::INFINITY {
                None
            } else {
                Some((self.distances[g], self.reconstruct_path(source, g)))
            }
        } else {
            // This branch is for returning all distances.
            // The path is empty because it's not a single-path query.
            Some((0.0, Vec::new()))
        }
    }

    /// The recursive core of the algorithm.
    /// `bmssp` stands for "Breaking the Mold Single-Source Shortest Path".
    fn bmssp(&mut self, level: usize, bound: f64, frontier: Vec<usize>) -> (f64, Vec<usize>) {
        if level == 0 {
            return self.base_case(bound, frontier);
        }

        let (pivots, working_set) = self.find_pivots(bound, &frontier);

        if working_set.len() > self.k * frontier.len() {
            // Early termination due to large working set
            return (bound, working_set);
        }

        let capacity = if level >= 1 {
            2_usize.pow(((level - 1) * self.t).min(20) as u32) // Cap to prevent overflow
        } else {
            1
        };
        let mut data_structure = AdaptiveDataStructure::new(capacity, bound);

        // Insert pivots into data structure
        pivots.iter().for_each(|&pivot| {
            if self.distances[pivot] != f64::INFINITY {
                data_structure.insert(pivot, self.distances[pivot]);
            }
        });

        let mut result_set = Vec::new();
        let mut current_bound = pivots
            .iter()
            .filter(|&&v| self.distances[v] != f64::INFINITY)
            .map(|&v| self.distances[v])
            .fold(f64::INFINITY, f64::min);

        let max_result_size = self.k * 2_usize.pow((level * self.t).min(20) as u32);

        // Main iteration loop
        while result_set.len() < max_result_size && !data_structure.is_empty() {
            let (subset_bound, subset) = data_structure.pull();

            if subset.is_empty() {
                break;
            }

            let (sub_bound, sub_result) = self.bmssp(level - 1, subset_bound, subset);
            result_set.extend(&sub_result);

            // Relax edges from newly completed vertices
            let mut batch_prepend_list = Vec::new();

            sub_result.iter().for_each(|&u| {
                if !self.complete.get(u).unwrap_or(false) {
                    self.complete.set(u, true);
                }

                self.graph.edges[u].iter().for_each(|edge| {
                    let v = edge.to;
                    let new_dist = self.distances[u] + edge.weight;

                    if new_dist < self.distances[v] {
                        self.distances[v] = new_dist;
                        self.predecessors[v] = Some(u);

                        if new_dist >= subset_bound && new_dist < bound {
                            data_structure.insert(v, new_dist);
                        } else if new_dist >= sub_bound && new_dist < subset_bound {
                            batch_prepend_list.push((v, new_dist));
                        }
                    }
                });
            });

            // Batch prepend operation
            data_structure.batch_prepend(batch_prepend_list);
            current_bound = current_bound.min(sub_bound);

            if result_set.len() >= max_result_size {
                break;
            }
        }

        // Add remaining complete vertices from working set
        working_set.iter().for_each(|&v| {
            if self.distances[v] < current_bound && !result_set.contains(&v) {
                result_set.push(v);
                self.complete.set(v, true);
            }
        });

        (current_bound, result_set)
    }

    fn base_case(&mut self, bound: f64, frontier: Vec<usize>) -> (f64, Vec<usize>) {
        if frontier.is_empty() {
            return (bound, Vec::new());
        }

        // Use Dijkstra-like approach from all frontier vertices
        let mut heap = BinaryHeap::new();
        let mut result = Vec::new();

        // Initialise heap with frontier vertices
        frontier.iter().for_each(|&start| {
            if self.distances[start] != f64::INFINITY {
                heap.push(Reverse(VertexDistance {
                    vertex: start,
                    distance: self.distances[start],
                }));
                result.push(start);
            }
        });

        let mut processed = 0;
        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            if dist > self.distances[u] || processed >= self.k + frontier.len() {
                continue;
            }

            self.complete.set(u, true);
            processed += 1;

            self.graph.edges[u].iter().for_each(|edge| {
                let v = edge.to;
                let new_dist = dist + edge.weight;

                if new_dist < bound && new_dist < self.distances[v] {
                    self.distances[v] = new_dist;
                    self.predecessors[v] = Some(u);
                    if !result.contains(&v) {
                        result.push(v);
                    }
                    heap.push(Reverse(VertexDistance {
                        vertex: v,
                        distance: new_dist,
                    }));
                }
            });
        }

        // Determine the boundary
        let max_dist = if result.len() > self.k + frontier.len() {
            let mut distances: Vec<f64> = result
                .iter()
                .map(|&v| self.distances[v])
                .filter(|&d| d != f64::INFINITY)
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if distances.len() > self.k {
                let boundary = distances[self.k];
                result.retain(|&v| self.distances[v] < boundary);
                boundary
            } else {
                bound
            }
        } else {
            bound
        };

        // Mark result vertices as complete
        result.iter().for_each(|&v| {
            if self.distances[v] != f64::INFINITY {
                self.complete.set(v, true);
            }
        });

        (max_dist, result)
    }

    fn find_pivots(&mut self, bound: f64, frontier: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut working_set: HashSet<usize> = frontier.iter().cloned().collect();
        let mut current_layer: HashSet<usize> = frontier.iter().cloned().collect();

        // Perform k relaxation steps
        for _ in 0..self.k {
            let mut next_layer = HashSet::new();

            for &u in &current_layer {
                if self.distances[u] == f64::INFINITY {
                    continue;
                }

                for edge in &self.graph.edges[u] {
                    let v = edge.to;
                    let new_dist = self.distances[u] + edge.weight;

                    if new_dist < self.distances[v] && new_dist < bound {
                        self.distances[v] = new_dist;
                        self.predecessors[v] = Some(u);
                        if !working_set.contains(&v) {
                            next_layer.insert(v);
                            working_set.insert(v);
                        }
                    }
                }
            }

            current_layer = next_layer;

            if working_set.len() > self.k * frontier.len() {
                return (frontier.to_vec(), working_set.into_iter().collect());
            }
        }

        // Find pivots (roots of large subtrees)
        let mut pivots = Vec::new();
        let mut subtree_sizes = HashMap::new();

        // Build forest structure based on predecessor relationships
        working_set.iter().for_each(|&v| {
            if let Some(pred) = self.predecessors[v] {
                *subtree_sizes.entry(pred).or_insert(0) += 1;
            }
        });

        // Select pivots with subtree size >= k, or just use frontier if no large subtrees
        subtree_sizes.iter().for_each(|(&root, &size)| {
            if size >= self.k && frontier.contains(&root) {
                pivots.push(root);
            }
        });

        // If no pivots found, use the entire frontier
        if pivots.is_empty() {
            pivots = frontier.to_vec();
        }

        (pivots, working_set.into_iter().collect())
    }
}

// Adaptive data structure for managing vertex priorities
struct AdaptiveDataStructure {
    data: BinaryHeap<Reverse<VertexDistance>>,
    capacity: usize,
    bound: f64,
}

impl AdaptiveDataStructure {
    fn new(capacity: usize, bound: f64) -> Self {
        AdaptiveDataStructure {
            data: BinaryHeap::new(),
            capacity,
            bound,
        }
    }

    fn insert(&mut self, vertex: usize, distance: f64) {
        if distance < self.bound {
            self.data.push(Reverse(VertexDistance { vertex, distance }));
        }
    }

    fn batch_prepend(&mut self, items: Vec<(usize, f64)>) {
        items.into_iter().for_each(|(vertex, distance)| {
            self.insert(vertex, distance);
        });
    }

    fn pull(&mut self) -> (f64, Vec<usize>) {
        let mut result = Vec::new();
        let mut min_remaining = self.bound;

        while result.len() < self.capacity && !self.data.is_empty() {
            if let Some(Reverse(VertexDistance {
                vertex,
                distance: _,
            })) = self.data.pop()
            {
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

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// Parallel stubs
#[cfg(feature = "parallel_pivot")]
mod parallel_pivot {
    // pub fn par_find_pivots(...)
    // todo!()
}

#[cfg(feature = "parallel_frontier_expansion")]
mod parallel_frontier_expansion {
    // pub fn par_find_frontier(...)
    // todo!()
}

#[cfg(feature = "parallel_edge_relaxation")]
mod parallel_edge_relaxation {
    // pub fn par_relax_edges(...)
    // todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_graph() {
        // Create a larger graph (15 vertices), because on <=10 we run Dijkstra per the paper's recommendations.
        let mut graph = Graph::new(15);

        for i in 0..14 {
            graph.add_edge(i, i + 1, (i + 1) as f64);
            if i > 0 {
                graph.add_edge(i, i - 1, (i as f64) * 0.5);
            }
        }
        // Add some cross edges
        graph.add_edge(0, 5, 10.0);
        graph.add_edge(2, 8, 15.0);
        graph.add_edge(7, 12, 8.0);

        let result = dun_mao_shortest_path(&graph, 0, 14);

        assert!(result.is_some());
        let (distance, path) = result.unwrap();
        assert!(distance > 0.0 && distance < f64::INFINITY);
        assert_eq!(path.first(), Some(&0));
        assert_eq!(path.last(), Some(&14));
    }

    #[test]
    fn disconnected_graph() {
        let mut graph = Graph::new(20);

        // Connected component 1: vertices 0-9
        for i in 0..9 {
            graph.add_edge(i, i + 1, 2.0);
        }

        // Connected component 2: vertices 10-19 (disconnected from 0)
        for i in 10..19 {
            graph.add_edge(i, i + 1, 3.0);
        }

        let result = dun_mao_shortest_path(&graph, 0, 15);

        assert!(result.is_none());
    }

    #[test]
    fn single_vertex() {
        let graph = Graph::new(1);
        let result = dun_mao_shortest_path(&graph, 0, 0);

        assert!(result.is_some());
        let (distance, path) = result.unwrap();
        assert_eq!(distance, 0.0);
        assert_eq!(path, vec![0]);
    }

    #[test]
    fn algorithm_comparison() {
        // Test that both algorithms give same results on same graph
        let mut graph = Graph::new(12);

        // Create a path graph with some shortcuts
        for i in 0..11 {
            graph.add_edge(i, i + 1, 1.0);
        }
        graph.add_edge(0, 5, 3.0);
        graph.add_edge(2, 8, 4.0);

        // Test with new algorithm
        let result1 = dun_mao_shortest_path(&graph, 0, 11);

        // Test with Dijkstra
        let mut solver2 = SSSpSolver::new(graph.clone());
        let result2 = solver2.dijkstra(0, Some(11));

        assert!(result1.is_some());
        assert!(result2.is_some());

        let (distance1, _) = result1.unwrap();
        let (distance2, _) = result2.unwrap();

        assert!((distance1 - distance2).abs() < 1e-10);
    }
}
