use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam::channel;
#[cfg(feature = "hashbrown")]
use hashbrown::HashSet;
#[cfg(not(feature = "hashbrown"))]
use std::collections::HashSet;

use crate::graph::Graph;
use crate::parallel::atomic_f64::AtomicF64;
use crate::utils::{AdaptiveDataStructure, INFINITY, VertexDistance};

pub mod atomic_f64;

pub struct ParallelSSSpSolver {
    graph: Arc<Graph>,
    distances: Arc<Vec<AtomicF64>>,
    predecessors: Arc<Vec<AtomicUsize>>,
    complete: Arc<Vec<AtomicBool>>,
    k: usize,
    t: usize,
    num_threads: usize,
}

impl ParallelSSSpSolver {
    pub fn new(graph: Graph, num_threads: usize) -> Self {
        let n = graph.vertices;
        let k = ((n as f64).ln().powf(1.0 / 3.0)).floor() as usize;
        let t = ((n as f64).ln().powf(2.0 / 3.0)).floor() as usize;

        let mut distances = Vec::with_capacity(n);
        let mut predecessors = Vec::with_capacity(n);
        let mut complete = Vec::with_capacity(n);

        for _ in 0..n {
            distances.push(AtomicF64::new(INFINITY));
            predecessors.push(AtomicUsize::new(usize::MAX));
            complete.push(AtomicBool::new(false));
        }

        Self {
            graph: Arc::new(graph),
            distances: Arc::new(distances),
            predecessors: Arc::new(predecessors),
            complete: Arc::new(complete),
            k: k.max(3),
            t: t.max(2),
            num_threads,
        }
    }

    pub fn solve(&self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        self.solve_all(source);
        if self.distances[goal].load(Ordering::Relaxed) == INFINITY {
            None
        } else {
            Some((
                self.distances[goal].load(Ordering::Relaxed),
                self.reconstruct_path(source, goal),
            ))
        }
    }

    pub fn solve_all(&self, source: usize) -> HashMap<usize, f64> {
        // Reset state
        self.reset_state();

        self.distances[source].store(0.0, Ordering::Relaxed);

        // For small graphs, use sequential algorithm
        if self.graph.vertices <= 50 {
            return self.sequential_fallback(source);
        }

        let max_level = ((self.graph.vertices as f64).ln() / self.t as f64).ceil() as usize;
        let frontier = vec![source];

        let (_bound, _result) = self.parallel_bmssp(max_level, INFINITY, frontier);

        // Complete any remaining vertices with Dijkstra
        self.complete_with_dijkstra();

        let mut results = HashMap::new();
        for i in 0..self.graph.vertices {
            let dist = self.distances[i].load(Ordering::Relaxed);
            if dist != INFINITY {
                results.insert(i, dist);
            }
        }
        results
    }

    fn reset_state(&self) {
        for i in 0..self.graph.vertices {
            self.distances[i].store(INFINITY, Ordering::Relaxed);
            self.predecessors[i].store(usize::MAX, Ordering::Relaxed);
            self.complete[i].store(false, Ordering::Relaxed);
        }
    }

    fn sequential_fallback(&self, source: usize) -> HashMap<usize, f64> {
        // Use a simple Dijkstra for small graphs
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(VertexDistance {
            vertex: source,
            distance: 0.0,
        }));

        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            let current_dist = self.distances[u].load(Ordering::Relaxed);
            if current_dist < dist {
                continue;
            }

            if self.complete[u].load(Ordering::Relaxed) {
                continue;
            }

            self.complete[u].store(true, Ordering::Relaxed);

            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = dist + edge.weight;
                let old_dist = self.distances[v].load(Ordering::Relaxed);

                if new_dist < old_dist {
                    self.distances[v].store(new_dist, Ordering::Relaxed);
                    self.predecessors[v].store(u, Ordering::Relaxed);
                    heap.push(Reverse(VertexDistance {
                        vertex: v,
                        distance: new_dist,
                    }));
                }
            }
        }

        let mut results = HashMap::new();
        for i in 0..self.graph.vertices {
            let dist = self.distances[i].load(Ordering::Relaxed);
            if dist != INFINITY {
                results.insert(i, dist);
            }
        }
        results
    }

    fn parallel_bmssp(&self, level: usize, bound: f64, frontier: Vec<usize>) -> (f64, Vec<usize>) {
        if level == 0 || frontier.len() <= 1 {
            return self.sequential_base_case(bound, frontier);
        }

        let (pivots, working_set) = self.parallel_find_pivots(bound, &frontier);

        if working_set.len() > self.k * frontier.len() {
            return (bound, working_set);
        }

        // For deeper levels, fall back to sequential to avoid thread explosion
        if level <= 2 || pivots.len() < self.num_threads {
            return self.sequential_bmssp(level, bound, pivots);
        }

        let chunk_size = pivots.len().div_ceil(self.num_threads);
        let pivot_chunks: Vec<_> = pivots.chunks(chunk_size).collect();

        let (sender, receiver) = channel::unbounded();
        let mut handles = Vec::new();

        for chunk in pivot_chunks {
            if chunk.is_empty() {
                continue;
            }
            let chunk = chunk.to_vec();
            let sender = sender.clone();
            let solver = self.clone_for_thread();

            let handle = thread::spawn(move || {
                let result = solver.sequential_bmssp(level - 1, bound, chunk);
                sender.send(result).unwrap();
            });
            handles.push(handle);
        }

        drop(sender);

        let mut combined_result = Vec::new();
        let mut min_boundary = bound;

        for result in receiver {
            let (boundary, vertices) = result;
            combined_result.extend(vertices);
            min_boundary = min_boundary.min(boundary);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        (min_boundary, combined_result)
    }

    fn sequential_bmssp(&self, level: usize, bound: f64, pivots: Vec<usize>) -> (f64, Vec<usize>) {
        if level == 0 {
            return self.sequential_base_case(bound, pivots);
        }

        let mut data_structure =
            AdaptiveDataStructure::new(2_usize.pow(((level * self.t).min(20)) as u32), bound);

        for &pivot in &pivots {
            let dist = self.distances[pivot].load(Ordering::Relaxed);
            if dist != INFINITY {
                data_structure.insert(pivot, dist);
            }
        }

        let mut result_set = Vec::new();
        let max_result_size = self.k * 2_usize.pow((level * self.t).min(20) as u32);

        while result_set.len() < max_result_size && !data_structure.is_empty() {
            let (subset_bound, subset) = data_structure.pull();

            if subset.is_empty() {
                break;
            }

            let (_sub_bound, sub_result) = self.sequential_bmssp(level - 1, subset_bound, subset);

            for &vertex in &sub_result {
                self.complete[vertex].store(true, Ordering::Relaxed);
                result_set.push(vertex);
            }

            self.sequential_edge_relaxation(&sub_result, subset_bound, bound, &mut data_structure);
        }

        (bound, result_set)
    }

    fn sequential_base_case(&self, bound: f64, frontier: Vec<usize>) -> (f64, Vec<usize>) {
        if frontier.is_empty() {
            return (bound, Vec::new());
        }

        let mut heap = BinaryHeap::new();
        let mut result = HashSet::new();

        for &start in &frontier {
            let dist = self.distances[start].load(Ordering::Relaxed);
            if dist != INFINITY {
                heap.push(Reverse(VertexDistance {
                    vertex: start,
                    distance: dist,
                }));
                result.insert(start);
            }
        }

        let mut processed = 0;
        let max_process = self.k + frontier.len();

        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            let current_dist = self.distances[u].load(Ordering::Relaxed);
            if dist > current_dist || processed >= max_process {
                continue;
            }

            if self.complete[u].load(Ordering::Relaxed) {
                continue;
            }

            self.complete[u].store(true, Ordering::Relaxed);
            processed += 1;

            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = dist + edge.weight;

                if new_dist < bound {
                    let old_dist = self.distances[v].load(Ordering::Relaxed);
                    if new_dist < old_dist {
                        self.distances[v].store(new_dist, Ordering::Relaxed);
                        self.predecessors[v].store(u, Ordering::Relaxed);
                        result.insert(v);

                        heap.push(Reverse(VertexDistance {
                            vertex: v,
                            distance: new_dist,
                        }));
                    }
                }
            }
        }

        let mut final_result: Vec<usize> = result.into_iter().collect();

        let boundary = if final_result.len() > self.k + frontier.len() {
            let mut distances: Vec<f64> = final_result
                .iter()
                .map(|&v| self.distances[v].load(Ordering::Relaxed))
                .filter(|d| *d != INFINITY)
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            if distances.len() > self.k {
                let boundary = distances[self.k];
                final_result.retain(|&v| self.distances[v].load(Ordering::Relaxed) < boundary);
                boundary
            } else {
                bound
            }
        } else {
            bound
        };

        for &v in &final_result {
            if self.distances[v].load(Ordering::Relaxed) != INFINITY {
                self.complete[v].store(true, Ordering::Relaxed);
            }
        }

        (boundary, final_result)
    }

    fn parallel_find_pivots(&self, bound: f64, frontier: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let working_set: Arc<Mutex<HashSet<usize>>> =
            Arc::new(Mutex::new(frontier.iter().copied().collect()));
        let mut current_layer: Vec<usize> = frontier.to_vec();

        for _step in 0..self.k {
            if current_layer.is_empty() {
                break;
            }

            let next_layer = Arc::new(Mutex::new(HashSet::new()));
            let updates = Arc::new(Mutex::new(Vec::new()));

            // Only parallelize if we have enough work
            if current_layer.len() < self.num_threads * 4 {
                // Process sequentially for small layers
                let mut local_updates = Vec::new();
                let mut local_next_layer = HashSet::new();

                for &u in &current_layer {
                    let u_dist = self.distances[u].load(Ordering::Relaxed);
                    if u_dist == INFINITY {
                        continue;
                    }

                    for edge in &self.graph.edges[u] {
                        let v = edge.to;
                        let new_dist = u_dist + edge.weight;

                        if new_dist < bound {
                            let old_dist = self.distances[v].load(Ordering::Relaxed);
                            if new_dist < old_dist {
                                local_updates.push((v, new_dist, u));
                                let working_guard = working_set.lock().unwrap();
                                if !working_guard.contains(&v) {
                                    local_next_layer.insert(v);
                                }
                            }
                        }
                    }
                }

                // Apply updates
                for (vertex, new_dist, predecessor) in local_updates {
                    self.atomic_distance_update(vertex, new_dist, predecessor);
                }

                // Update working set and current layer
                {
                    let mut working_guard = working_set.lock().unwrap();
                    working_guard.extend(&local_next_layer);
                    current_layer = local_next_layer.into_iter().collect();
                }
            } else {
                // Process in parallel
                let chunk_size = current_layer.len().div_ceil(self.num_threads);
                let layer_chunks: Vec<_> = current_layer.chunks(chunk_size).collect();

                let mut handles = Vec::new();

                for chunk in layer_chunks {
                    if chunk.is_empty() {
                        continue;
                    }
                    let chunk = chunk.to_vec();
                    let next_layer = Arc::clone(&next_layer);
                    let updates = Arc::clone(&updates);
                    let working_set = Arc::clone(&working_set);
                    let graph = Arc::clone(&self.graph);
                    let distances = Arc::clone(&self.distances);

                    let handle = thread::spawn(move || {
                        let mut local_updates = Vec::new();
                        let mut local_next_layer = HashSet::new();

                        for &u in &chunk {
                            let u_dist = distances[u].load(Ordering::Relaxed);
                            if u_dist == INFINITY {
                                continue;
                            }

                            for edge in &graph.edges[u] {
                                let v = edge.to;
                                let new_dist = u_dist + edge.weight;

                                if new_dist < bound {
                                    let old_dist = distances[v].load(Ordering::Relaxed);
                                    if new_dist < old_dist {
                                        local_updates.push((v, new_dist, u));
                                        let working_guard = working_set.lock().unwrap();
                                        if !working_guard.contains(&v) {
                                            local_next_layer.insert(v);
                                        }
                                    }
                                }
                            }
                        }

                        updates.lock().unwrap().extend(local_updates);
                        next_layer.lock().unwrap().extend(local_next_layer);
                    });

                    handles.push(handle);
                }

                for handle in handles {
                    handle.join().unwrap();
                }

                // Apply all updates atomically
                let updates_guard = updates.lock().unwrap();
                for &(vertex, new_dist, predecessor) in updates_guard.iter() {
                    self.atomic_distance_update(vertex, new_dist, predecessor);
                }

                // Update working set and current layer
                {
                    let mut working_guard = working_set.lock().unwrap();
                    let next_guard = next_layer.lock().unwrap();
                    working_guard.extend(next_guard.iter());
                    current_layer = next_guard.iter().copied().collect();
                }
            }

            // Early termination check
            let working_size = working_set.lock().unwrap().len();
            if working_size > self.k * frontier.len() {
                break;
            }
        }

        let working_vec: Vec<usize> = working_set.lock().unwrap().iter().copied().collect();
        let pivots = self.extract_pivots(&working_vec, frontier);

        (pivots, working_vec)
    }

    fn atomic_distance_update(&self, vertex: usize, new_dist: f64, predecessor: usize) {
        let mut old_dist = self.distances[vertex].load(Ordering::Acquire);

        loop {
            if new_dist >= old_dist {
                break; // Another thread found a better path
            }

            match self.distances[vertex].compare_exchange_weak(
                old_dist,
                new_dist,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.predecessors[vertex].store(predecessor, Ordering::Relaxed);
                    break;
                }
                Err(current) => {
                    old_dist = current;
                }
            }
        }
    }

    fn sequential_edge_relaxation(
        &self,
        completed_vertices: &[usize],
        lower_bound: f64,
        upper_bound: f64,
        data_structure: &mut AdaptiveDataStructure,
    ) {
        let mut batch_prepend_items = Vec::new();

        for &u in completed_vertices {
            let u_dist = self.distances[u].load(Ordering::Relaxed);
            if u_dist == INFINITY {
                continue;
            }

            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = u_dist + edge.weight;
                let old_dist = self.distances[v].load(Ordering::Relaxed);

                if new_dist < old_dist {
                    self.distances[v].store(new_dist, Ordering::Relaxed);
                    self.predecessors[v].store(u, Ordering::Relaxed);

                    if new_dist >= lower_bound && new_dist < upper_bound {
                        data_structure.insert(v, new_dist);
                    } else if new_dist < lower_bound {
                        batch_prepend_items.push((v, new_dist));
                    }
                }
            }
        }

        if !batch_prepend_items.is_empty() {
            data_structure.batch_prepend(batch_prepend_items);
        }
    }

    fn complete_with_dijkstra(&self) {
        let mut heap = BinaryHeap::new();

        // Add all incomplete vertices with finite distances
        for i in 0..self.graph.vertices {
            if !self.complete[i].load(Ordering::Relaxed) {
                let dist = self.distances[i].load(Ordering::Relaxed);
                if dist != INFINITY {
                    heap.push(Reverse(VertexDistance {
                        vertex: i,
                        distance: dist,
                    }));
                }
            }
        }

        while let Some(Reverse(VertexDistance {
            vertex: u,
            distance: dist,
        })) = heap.pop()
        {
            if self.complete[u].load(Ordering::Relaxed) {
                continue;
            }

            let current_dist = self.distances[u].load(Ordering::Relaxed);
            if dist > current_dist {
                continue;
            }

            self.complete[u].store(true, Ordering::Relaxed);

            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = dist + edge.weight;
                let old_dist = self.distances[v].load(Ordering::Relaxed);

                if new_dist < old_dist {
                    self.distances[v].store(new_dist, Ordering::Relaxed);
                    self.predecessors[v].store(u, Ordering::Relaxed);

                    if !self.complete[v].load(Ordering::Relaxed) {
                        heap.push(Reverse(VertexDistance {
                            vertex: v,
                            distance: new_dist,
                        }));
                    }
                }
            }
        }
    }

    fn clone_for_thread(&self) -> Self {
        Self {
            graph: Arc::clone(&self.graph),
            distances: Arc::clone(&self.distances),
            predecessors: Arc::clone(&self.predecessors),
            complete: Arc::clone(&self.complete),
            k: self.k,
            t: self.t,
            num_threads: self.num_threads,
        }
    }

    fn extract_pivots(&self, working_vec: &[usize], frontier: &[usize]) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut subtree_sizes = HashMap::new();

        for &v in working_vec {
            let pred = self.predecessors[v].load(Ordering::Relaxed);
            if pred != usize::MAX {
                *subtree_sizes.entry(pred).or_insert(0) += 1;
            }
        }

        for (&root, &size) in &subtree_sizes {
            if size >= self.k && frontier.contains(&root) {
                pivots.push(root);
            }
        }

        if pivots.is_empty() {
            frontier.to_vec()
        } else {
            pivots
        }
    }

    fn reconstruct_path(&self, source: usize, goal: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = goal;

        while current != source {
            path.push(current);
            let pred = self.predecessors[current].load(Ordering::Relaxed);
            if pred != usize::MAX {
                current = pred;
            } else {
                return Vec::new();
            }
        }

        path.push(source);
        path.reverse();
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::sequential::SSSpSolver as SequentialSolver;
    use std::time::Instant;

    fn create_test_graph(size: usize) -> Graph {
        let mut graph = Graph::new(size);
        for i in 0..size - 1 {
            graph.add_edge(i, i + 1, 1.0);
        }
        graph
    }

    fn create_large_test_graph(size: usize) -> Graph {
        let mut graph = Graph::new(size);
        for i in 0..size {
            for j in 0..3 {
                graph.add_edge(i, (i + j + 1) % size, 1.0);
            }
        }
        graph
    }

    #[test]
    fn test_parallel_correctness() {
        let graph = create_test_graph(1000);

        let mut seq_solver = SequentialSolver::new(graph.clone());
        let seq_distances = seq_solver.solve_all(0);

        let par_solver = ParallelSSSpSolver::new(graph, 4);
        let par_distances = par_solver.solve_all(0);

        assert_eq!(seq_distances.len(), par_distances.len());
        for (vertex, &seq_dist) in &seq_distances {
            let par_dist = par_distances.get(vertex).unwrap();
            assert!((seq_dist - par_dist).abs() < 1e-10);
        }
    }

    #[test]
    fn benchmark_parallel_speedup() {
        let graph = create_large_test_graph(10000);

        let start = Instant::now();
        let mut seq_solver = SequentialSolver::new(graph.clone());
        seq_solver.solve_all(0);
        let seq_time = start.elapsed();

        for threads in [1, 2, 4, 8] {
            let start = Instant::now();
            let par_solver = ParallelSSSpSolver::new(graph.clone(), threads);
            par_solver.solve_all(0);
            let par_time = start.elapsed();

            let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64();
            println!("Threads: {}, Speedup: {:.2}x", threads, speedup);
        }
    }
}
