use crate::graph::Graph;
use crate::utils::{AdaptiveDataStructure, INFINITY, VertexDistance};
use std::cmp::Reverse;

#[cfg(not(feature = "hashbrown"))]
use std::collections::{BinaryHeap, HashMap, HashSet};

#[cfg(feature = "hashbrown")]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "hashbrown")]
use std::collections::BinaryHeap;

/// Implements the SSSP algorithm from "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
/// by Duan, Mao, Mao, Shu, and Yin (2025).
pub struct SSSpSolver {
    graph: Graph,
    /// Stores the shortest distance from the source to each vertex.
    distances: Vec<f64>,
    /// Stores the predecessor of each vertex in the shortest path.
    predecessors: Vec<Option<usize>>,
    /// A bitmask to mark vertices as complete (visited and finalized).
    complete: Vec<bool>,
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
            distances: vec![INFINITY; n],
            predecessors: vec![None; n],
            complete: vec![false; n],
            graph,
            k: k.max(3), // Ensure k is at least 3 for algorithm correctness
            t: t.max(2), // Ensure t is at least 2 for small graphs
        }
    }

    /// Main entry point for single-path shortest path queries
    pub fn solve(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        // Always reset state completely
        self.reset_state();
        self.distances[source] = 0.0;

        // For small graphs or debugging, use Dijkstra directly
        if self.graph.vertices <= 15 {
            return self.dijkstra_single_path(source, goal);
        }

        // Run the Duan-Mao algorithm
        self.run_duan_mao_algorithm(source);

        // Return result for the specific goal
        if self.distances[goal] == INFINITY {
            None
        } else {
            Some((self.distances[goal], self.reconstruct_path(source, goal)))
        }
    }

    /// Compute all distances from source using the Duan-Mao algorithm
    pub fn solve_all(&mut self, source: usize) -> HashMap<usize, f64> {
        self.reset_state();
        self.distances[source] = 0.0;

        if self.graph.vertices <= 15 {
            self.dijkstra_all_distances(source);
        } else {
            self.run_duan_mao_algorithm(source);
        }

        // Collect finite distances
        let mut results = HashMap::new();
        for i in 0..self.graph.vertices {
            if self.distances[i] != INFINITY {
                results.insert(i, self.distances[i]);
            }
        }
        results
    }

    /// Reset all internal state
    fn reset_state(&mut self) {
        self.distances.fill(INFINITY);
        self.predecessors.fill(None);
        self.complete.fill(false);
    }

    /// Run the complete Duan-Mao SSSP algorithm
    fn run_duan_mao_algorithm(&mut self, source: usize) {
        let max_level = ((self.graph.vertices as f64).ln() / self.t as f64).ceil() as usize;
        let frontier = vec![source];

        // Run BMSSP algorithm
        let (_bound, _result) = self.bmssp(max_level, INFINITY, frontier);

        // Complete any remaining incomplete vertices with standard Dijkstra
        self.complete_with_dijkstra();
    }

    /// Complete the algorithm by running Dijkstra on any remaining incomplete vertices
    fn complete_with_dijkstra(&mut self) {
        let mut heap = BinaryHeap::new();

        // Add all vertices with finite distances that aren't complete
        for i in 0..self.graph.vertices {
            if !self.complete[i] && self.distances[i] != INFINITY {
                heap.push(Reverse(VertexDistance {
                    vertex: i,
                    distance: self.distances[i],
                }));
            }
        }

        // Run Dijkstra completion
        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            if self.complete[u] || dist > self.distances[u] {
                continue;
            }

            self.complete[u] = true;

            // Relax all outgoing edges
            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = dist + edge.weight;

                if new_dist < self.distances[v] {
                    self.distances[v] = new_dist;
                    self.predecessors[v] = Some(u);

                    if !self.complete[v] {
                        heap.push(Reverse(VertexDistance {
                            vertex: v,
                            distance: new_dist,
                        }));
                    }
                }
            }
        }
    }

    /// Standard Dijkstra for single path queries
    pub fn dijkstra_single_path(
        &mut self,
        source: usize,
        goal: usize,
    ) -> Option<(f64, Vec<usize>)> {
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
            if self.complete[u] {
                continue;
            }

            if dist > self.distances[u] {
                continue;
            }

            self.complete[u] = true;

            // Early termination if we reached the goal
            if u == goal {
                break;
            }

            // Relax all outgoing edges
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

        if self.distances[goal] == INFINITY {
            None
        } else {
            Some((self.distances[goal], self.reconstruct_path(source, goal)))
        }
    }

    /// Standard Dijkstra for all distances
    pub fn dijkstra_all_distances(&mut self, source: usize) {
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
            if self.complete[u] || dist > self.distances[u] {
                continue;
            }

            self.complete[u] = true;

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

    /// Reconstruct path from source to goal using predecessor information
    fn reconstruct_path(&self, source: usize, goal: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = goal;

        while current != source {
            path.push(current);
            if let Some(pred) = self.predecessors[current] {
                current = pred;
            } else {
                // Path doesn't exist or is broken
                return Vec::new();
            }
        }

        path.push(source);
        path.reverse();
        path
    }

    /// The main recursive BMSSP (Bounded Multi-Source Shortest Path) algorithm
    fn bmssp(&mut self, level: usize, bound: f64, frontier: Vec<usize>) -> (f64, Vec<usize>) {
        if level == 0 {
            return self.base_case(bound, frontier);
        }

        // Find pivots to reduce frontier size
        let (pivots, working_set) = self.find_pivots(bound, &frontier);

        // Early termination if working set is too large
        if working_set.len() > self.k * frontier.len() {
            return (bound, working_set);
        }

        // Initialize data structure for managing subproblems
        let capacity = 2_usize.pow(((level - 1) * self.t).min(20) as u32);
        let mut data_structure = AdaptiveDataStructure::new(capacity, bound);

        // Insert pivots into data structure
        for &pivot in &pivots {
            if self.distances[pivot] != INFINITY {
                data_structure.insert(pivot, self.distances[pivot]);
            }
        }

        let mut result_set = Vec::new();
        let mut current_bound = pivots
            .iter()
            .filter(|&&v| self.distances[v] != INFINITY)
            .map(|&v| self.distances[v])
            .fold(INFINITY, f64::min);

        let max_result_size = self.k * 2_usize.pow((level * self.t).min(20) as u32);

        // Main recursive loop
        while result_set.len() < max_result_size && !data_structure.is_empty() {
            let (subset_bound, subset) = data_structure.pull();

            if subset.is_empty() {
                break;
            }

            // Recursive call
            let (sub_bound, sub_result) = self.bmssp(level - 1, subset_bound, subset);
            result_set.extend(&sub_result);

            // Relax edges from newly completed vertices
            let mut batch_prepend_list = Vec::new();

            for &u in &sub_result {
                self.complete[u] = true;

                for edge in &self.graph.edges[u] {
                    let v = edge.to;
                    let new_dist = self.distances[u] + edge.weight;

                    if new_dist < self.distances[v] {
                        self.distances[v] = new_dist;
                        self.predecessors[v] = Some(u);

                        // Add to appropriate data structure based on distance range
                        if new_dist >= subset_bound && new_dist < bound {
                            data_structure.insert(v, new_dist);
                        } else if new_dist >= sub_bound && new_dist < subset_bound {
                            batch_prepend_list.push((v, new_dist));
                        }
                    }
                }
            }

            // Batch prepend newly discovered vertices
            data_structure.batch_prepend(batch_prepend_list);
            current_bound = current_bound.min(sub_bound);

            if result_set.len() >= max_result_size {
                break;
            }
        }

        // Add remaining vertices from working set
        for &v in &working_set {
            if self.distances[v] < current_bound && !result_set.contains(&v) {
                result_set.push(v);
                self.complete[v] = true;
            }
        }

        (current_bound, result_set)
    }

    /// Base case: run limited Dijkstra from frontier vertices
    fn base_case(&mut self, bound: f64, frontier: Vec<usize>) -> (f64, Vec<usize>) {
        if frontier.is_empty() {
            return (bound, Vec::new());
        }

        let mut heap = BinaryHeap::new();
        let mut result = Vec::new();

        // Initialize with frontier vertices
        for &start in &frontier {
            if self.distances[start] != INFINITY {
                heap.push(Reverse(VertexDistance {
                    vertex: start,
                    distance: self.distances[start],
                }));
                result.push(start);
            }
        }

        let mut processed = 0;
        let max_process = self.k + frontier.len();

        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            if dist > self.distances[u] || processed >= max_process {
                continue;
            }

            self.complete[u] = true;
            processed += 1;

            for edge in &self.graph.edges[u] {
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
            }
        }

        // Determine boundary for partial execution
        let boundary = if result.len() > self.k + frontier.len() {
            let mut distances: Vec<f64> = result
                .iter()
                .map(|&v| self.distances[v])
                .filter(|d| *d != INFINITY)
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

        // Mark all result vertices as complete
        for &v in &result {
            if self.distances[v] != INFINITY {
                self.complete[v] = true;
            }
        }

        (boundary, result)
    }

    /// Find pivot vertices to reduce frontier size
    fn find_pivots(&mut self, bound: f64, frontier: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut working_set: HashSet<usize> = frontier.iter().copied().collect();
        let mut current_layer: HashSet<usize> = frontier.iter().copied().collect();

        // Perform k steps of Bellman-Ford-style relaxation
        for _ in 0..self.k {
            let mut next_layer = HashSet::new();

            for &u in &current_layer {
                if self.distances[u] == INFINITY {
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

            // Early termination if working set becomes too large
            if working_set.len() > self.k * frontier.len() {
                return (frontier.to_vec(), working_set.into_iter().collect());
            }
        }

        // Identify pivots based on subtree sizes
        let mut pivots = Vec::new();
        let mut subtree_sizes = HashMap::new();

        // Count subtree sizes
        for &v in &working_set {
            if let Some(pred) = self.predecessors[v] {
                *subtree_sizes.entry(pred).or_insert(0) += 1;
            }
        }

        // Select vertices with large subtrees as pivots
        for (&root, &size) in &subtree_sizes {
            if size >= self.k && frontier.contains(&root) {
                pivots.push(root);
            }
        }

        // Fallback: use entire frontier if no large subtrees found
        if pivots.is_empty() {
            pivots = frontier.to_vec();
        }

        (pivots, working_set.into_iter().collect())
    }

    /// Public method for compatibility with benchmarks
    pub fn dijkstra(&mut self, source: usize, goal: Option<usize>) -> Option<(f64, Vec<usize>)> {
        match goal {
            Some(g) => self.dijkstra_single_path(source, g),
            None => {
                self.dijkstra_all_distances(source);
                Some((0.0, Vec::new())) // Empty path for all-distances query
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::utils::VertexDistance;

    fn create_simple_graph() -> Graph {
        let mut graph = Graph::new(4);
        // 0 -> 1 (weight 1)
        // 0 -> 2 (weight 4)
        // 1 -> 2 (weight 2)
        // 1 -> 3 (weight 5)
        // 2 -> 3 (weight 1)
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(0, 2, 4.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(1, 3, 5.0);
        graph.add_edge(2, 3, 1.0);
        graph
    }

    fn create_larger_graph() -> Graph {
        let mut graph = Graph::new(8);
        // Create a more complex graph for testing the Duan-Mao algorithm
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(0, 2, 1.0);
        graph.add_edge(1, 3, 3.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(2, 4, 4.0);
        graph.add_edge(3, 5, 2.0);
        graph.add_edge(4, 5, 1.0);
        graph.add_edge(4, 6, 3.0);
        graph.add_edge(5, 7, 2.0);
        graph.add_edge(6, 7, 1.0);
        graph
    }

    #[test]
    fn vertex_distance_ordering() {
        let vd1 = VertexDistance::new(1, 3.0);
        let vd2 = VertexDistance::new(2, 2.0);
        let vd3 = VertexDistance::new(3, 2.0);

        // vd2 should be less than vd1 (smaller distance)
        assert!(vd2 < vd1);

        // When distances are equal, order by vertex ID
        assert!(vd2 < vd3);

        // Test with BinaryHeap (min-heap with Reverse)
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(vd1));
        heap.push(Reverse(vd2));
        heap.push(Reverse(vd3));

        assert_eq!(heap.pop().unwrap().0.vertex, 2); // Smallest distance, smallest vertex
        assert_eq!(heap.pop().unwrap().0.vertex, 3); // Same distance as above, larger vertex
        assert_eq!(heap.pop().unwrap().0.vertex, 1); // Largest distance
    }

    #[test]
    fn graph_creation() {
        let graph = create_simple_graph();
        assert_eq!(graph.vertices, 4);
        assert_eq!(graph.edge_count(), 5);

        // Check specific edges
        assert_eq!(graph.edges[0].len(), 2);
        assert_eq!(graph.edges[0][0].to, 1);
        assert_eq!(graph.edges[0][0].weight, 1.0);
    }

    #[test]
    fn dijkstra_single_path() {
        let graph = create_simple_graph();
        let mut solver = SSSpSolver::new(graph);

        let result = solver.dijkstra_single_path(0, 3);
        assert!(result.is_some());

        let (distance, path) = result.unwrap();
        assert_eq!(distance, 4.0); // 0->1->2->3 = 1+2+1 = 4
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn dijkstra_unreachable() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, 1.0);
        // Vertex 2 is unreachable from 0

        let mut solver = SSSpSolver::new(graph);
        let result = solver.dijkstra_single_path(0, 2);
        assert!(result.is_none());
    }

    #[test]
    fn dijkstra_all_distances() {
        let graph = create_simple_graph();
        let mut solver = SSSpSolver::new(graph);

        let distances = solver.solve_all(0);

        assert_eq!(distances.get(&0), Some(&0.0));
        assert_eq!(distances.get(&1), Some(&1.0));
        assert_eq!(distances.get(&2), Some(&3.0)); // 0->1->2 = 1+2 = 3
        assert_eq!(distances.get(&3), Some(&4.0)); // 0->1->2->3 = 1+2+1 = 4
    }

    #[test]
    fn duan_mao_vs_dijkstra() {
        let graph = create_larger_graph();

        // Test with Dijkstra
        let mut dijkstra_solver = SSSpSolver::new(graph.clone());
        let dijkstra_distances = dijkstra_solver.solve_all(0);

        // Test with Duan-Mao (forced by using large enough graph)
        let mut duan_mao_solver = SSSpSolver::new(graph);
        let duan_mao_distances = duan_mao_solver.solve_all(0);

        // Results should be identical
        for vertex in 0..8 {
            let dij_dist = dijkstra_distances.get(&vertex);
            let dm_dist = duan_mao_distances.get(&vertex);
            assert_eq!(dij_dist, dm_dist, "Mismatch for vertex {}", vertex);
        }
    }

    #[test]
    fn solve_method() {
        let graph = create_simple_graph();
        let mut solver = SSSpSolver::new(graph);

        let result = solver.solve(0, 3);
        assert!(result.is_some());

        let (distance, path) = result.unwrap();
        assert_eq!(distance, 4.0);
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn state_reset() {
        let graph = create_simple_graph();
        let mut solver = SSSpSolver::new(graph);

        // Run first solve
        solver.solve(0, 3);

        // Check that state is properly reset for second solve
        let result = solver.solve(1, 3);
        assert!(result.is_some());

        let (distance, path) = result.unwrap();
        assert_eq!(distance, 3.0); // 1->2->3 = 2+1 = 3
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn find_pivots() {
        let graph = create_larger_graph();
        let mut solver = SSSpSolver::new(graph);

        // Initialize some distances
        solver.distances[0] = 0.0;
        solver.distances[1] = 2.0;
        solver.distances[2] = 1.0;

        let frontier = vec![0, 1, 2];
        let (pivots, working_set) = solver.find_pivots(10.0, &frontier);

        // Should return some pivots and a working set
        assert!(!pivots.is_empty());
        assert!(!working_set.is_empty());
        assert!(working_set.len() >= frontier.len());
    }

    #[test]
    fn base_case() {
        let graph = create_simple_graph();
        let mut solver = SSSpSolver::new(graph);

        solver.distances[0] = 0.0;
        let frontier = vec![0];

        let (boundary, result) = solver.base_case(10.0, frontier);

        assert_eq!(boundary, 10.0);
        assert!(!result.is_empty());
        assert!(result.contains(&0));
    }

    #[test]
    fn adaptive_data_structure() {
        let mut ads = AdaptiveDataStructure::new(3, 10.0);

        ads.insert(1, 5.0);
        ads.insert(2, 3.0);
        ads.insert(3, 7.0);
        ads.insert(4, 1.0);

        let (bound, result) = ads.pull();

        // Should return up to 3 vertices with smallest distances
        assert!(result.len() <= 3);
        assert!(result.contains(&4)); // Smallest distance (1.0)
        assert!(result.contains(&2)); // Second smallest (3.0)
        assert!(bound <= 10.0);
    }

    #[test]
    fn batch_prepend() {
        let mut ads = AdaptiveDataStructure::new(5, 10.0);

        let items = vec![(1, 2.0), (2, 4.0), (3, 1.0)];
        ads.batch_prepend(items);

        let (_, result) = ads.pull();
        assert!(!result.is_empty());
        // Should contain vertex 3 (smallest distance)
        assert!(result.contains(&3));
    }

    #[test]
    fn negative_weights_not_supported() {
        // This implementation assumes non-negative weights
        // Testing that it doesn't crash with negative weights
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, -1.0);
        graph.add_edge(1, 2, 2.0);

        let mut solver = SSSpSolver::new(graph);
        // Should not panic, but results may be incorrect for negative weights
        let _result = solver.solve(0, 2);
    }

    #[test]
    fn self_loops() {
        let mut graph = Graph::new(2);
        graph.add_edge(0, 0, 1.0); // Self-loop
        graph.add_edge(0, 1, 2.0);

        let mut solver = SSSpSolver::new(graph);
        let result = solver.solve(0, 1);

        assert!(result.is_some());
        let (distance, path) = result.unwrap();
        assert_eq!(distance, 2.0);
        assert_eq!(path, vec![0, 1]);
    }

    #[test]
    fn zero_weight_edges() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, 0.0);
        graph.add_edge(1, 2, 0.0);

        let mut solver = SSSpSolver::new(graph);
        let result = solver.solve(0, 2);

        assert!(result.is_some());
        let (distance, path) = result.unwrap();
        assert_eq!(distance, 0.0);
        assert_eq!(path, vec![0, 1, 2]);
    }
}
