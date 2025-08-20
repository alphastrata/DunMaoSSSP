use std::collections::HashMap;
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
use crate::utils::{AdaptiveDataStructure, INFINITY};
use crossbeam::deque::{Injector, Steal, Stealer, Worker};

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
    /// Parallel BMSSP with pivot-level parallelization
    fn parallel_bmssp(
        &self,
        level: usize,
        bound: f64,
        pivots: Vec<usize>,
    ) -> (f64, Vec<usize>) {
        if level == 0 {
            // NOTE: This should be a parallel base case, but using sequential for now
            // return self.parallel_base_case(bound, pivots);
            return (bound, pivots); // Placeholder
        }

        // Partition pivots among threads
        let chunk_size = (pivots.len() + self.num_threads - 1) / self.num_threads;
        let pivot_chunks: Vec<_> = pivots.chunks(chunk_size).collect();

        let (sender, receiver) = channel::unbounded();
        let mut handles = Vec::new();

        let self_arc = Arc::new(self.clone_for_thread());

        // Spawn worker threads for each pivot chunk
        for chunk in pivot_chunks {
            let chunk = chunk.to_vec();
            let sender = sender.clone();
            let solver = Arc::clone(&self_arc);

            let handle = thread::spawn(move || {
                let result = solver.process_pivot_chunk(level - 1, bound, chunk);
                sender.send(result).unwrap();
            });
            handles.push(handle);
        }

        drop(sender); // Close the sender

        // Collect results from all threads
        let mut combined_result = Vec::new();
        let mut min_boundary = bound;

        for result in receiver {
            let (boundary, vertices) = result;
            combined_result.extend(vertices);
            min_boundary = min_boundary.min(boundary);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        (min_boundary, combined_result)
    }

    fn process_pivot_chunk(
        &self,
        level: usize,
        bound: f64,
        pivots: Vec<usize>,
    ) -> (f64, Vec<usize>) {
        // Initialize thread-local data structure
        let mut local_data_structure =
            AdaptiveDataStructure::new(2_usize.pow(((level * self.t).min(20)) as u32), bound);

        // Insert pivots with atomic distance reads
        for &pivot in &pivots {
            let dist = self.distances[pivot].load(Ordering::Relaxed);
            if dist != INFINITY {
                local_data_structure.insert(pivot, dist);
            }
        }

        let mut result_set = Vec::new();
        let max_result_size = self.k * 2_usize.pow((level * self.t).min(20) as u32);

        // Process subproblems sequentially within this thread
        while result_set.len() < max_result_size && !local_data_structure.is_empty() {
            let (subset_bound, subset) = local_data_structure.pull();

            if subset.is_empty() {
                break;
            }

            // Recursive call (could be further parallelized)
            // This is a conceptual placeholder for the sequential bmssp logic
            // let (sub_bound, sub_result) = self.sequential_bmssp(level, subset_bound, subset);
            let (_sub_bound, sub_result) = (subset_bound, subset.clone()); // Placeholder

            // Atomically update global state
            for &vertex in &sub_result {
                self.complete[vertex].store(true, Ordering::Relaxed);
                result_set.push(vertex);
            }

            // Parallel edge relaxation for newly completed vertices
            self.parallel_edge_relaxation(
                &sub_result,
                subset_bound,
                bound,
                &mut local_data_structure,
            );
        }

        (bound, result_set)
    }

    /// Parallel pivot finding with layer-wise parallelization
    fn parallel_find_pivots(
        &self,
        bound: f64,
        frontier: &[usize],
    ) -> (Vec<usize>, Vec<usize>) {
        let working_set: Arc<Mutex<HashSet<usize>>> =
            Arc::new(Mutex::new(frontier.iter().copied().collect()));
        let mut current_layer: Vec<usize> = frontier.to_vec();

        // Perform k steps of parallel relaxation
        for _ in 0..self.k {
            let next_layer = Arc::new(Mutex::new(HashSet::new()));
            let updates = Arc::new(Mutex::new(Vec::new()));

            // Partition current layer among threads
            let chunk_size = (current_layer.len() + self.num_threads - 1) / self.num_threads;
            let layer_chunks: Vec<_> = current_layer.chunks(chunk_size).collect();

            let mut handles = Vec::new();

            for chunk in layer_chunks {
                let chunk = chunk.to_vec();
                let next_layer = Arc::clone(&next_layer);
                let updates = Arc::clone(&updates);
                let working_set = Arc::clone(&working_set);
                let graph = Arc::clone(&self.graph);
                let distances = Arc::clone(&self.distances);

                let handle = thread::spawn(move || {
                    let mut local_updates = Vec::new();
                    let mut local_next_layer = HashSet::new();

                    // Process edges for vertices in this chunk
                    for &u in &chunk {
                        let u_dist = distances[u].load(Ordering::Acquire);
                        if u_dist == INFINITY {
                            continue;
                        }

                        for edge in &graph.edges[u] {
                            let v = edge.to;
                            let new_dist = u_dist + edge.weight;

                            if new_dist < bound {
                                let old_dist = distances[v].load(Ordering::Acquire);

                                // Attempt atomic update if improvement found
                                if new_dist < old_dist {
                                    local_updates.push((v, new_dist, u));

                                    // Check if vertex should be added to working set
                                    let working_guard = working_set.lock().unwrap();
                                    if !working_guard.contains(&v) {
                                        local_next_layer.insert(v);
                                    }
                                }
                            }
                        }
                    }

                    // Merge local results
                    updates.lock().unwrap().extend(local_updates);
                    next_layer.lock().unwrap().extend(local_next_layer);
                });

                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }

            // Apply all updates atomically
            let updates_guard = updates.lock().unwrap();
            for &(vertex, new_dist, predecessor) in updates_guard.iter() {
                // Use compare-and-swap for atomic distance updates
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
                            // Successfully updated distance
                            self.predecessors[vertex].store(predecessor, Ordering::Relaxed);
                            break;
                        }
                        Err(current) => {
                            old_dist = current;
                            // Retry with current value
                        }
                    }
                }
            }

            // Update working set and current layer
            let mut working_guard = working_set.lock().unwrap();
            let next_guard = next_layer.lock().unwrap();
            working_guard.extend(next_guard.iter());
            current_layer = next_guard.iter().copied().collect();

            // Early termination check
            if working_guard.len() > self.k * frontier.len() {
                break;
            }
        }

        // Extract pivots based on subtree sizes (sequential for now)
        let working_vec: Vec<usize> = working_set.lock().unwrap().iter().copied().collect();
        let pivots = self.extract_pivots(&working_vec, frontier);

        (pivots, working_vec)
    }

    /// Parallel edge relaxation with lock-free updates
    fn parallel_edge_relaxation(
        &self,
        completed_vertices: &[usize],
        lower_bound: f64,
        upper_bound: f64,
        data_structure: &mut AdaptiveDataStructure,
    ) {
        // Collect all edges to process
        let mut all_edges = Vec::new();
        for &u in completed_vertices {
            let u_dist = self.distances[u].load(Ordering::Acquire);
            for edge in &self.graph.edges[u] {
                all_edges.push((u, edge.to, edge.weight, u_dist));
            }
        }

        // Partition edges among threads
        let chunk_size = (all_edges.len() + self.num_threads - 1) / self.num_threads;
        let edge_chunks: Vec<_> = all_edges.chunks(chunk_size).collect();

        let updates = Arc::new(Mutex::new(Vec::new()));
        let batch_prepend_items = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::new();

        for chunk in edge_chunks {
            let chunk = chunk.to_vec();
            let updates = Arc::clone(&updates);
            let batch_prepend_items = Arc::clone(&batch_prepend_items);
            let distances = Arc::clone(&self.distances);
            let predecessors = Arc::clone(&self.predecessors);

            let handle = thread::spawn(move || {
                let mut local_updates = Vec::new();
                let mut local_batch_items = Vec::new();

                for (u, v, weight, u_dist) in chunk {
                    if u_dist == INFINITY {
                        continue;
                    }

                    let new_dist = u_dist + weight;
                    let old_dist = distances[v].load(Ordering::Acquire);

                    // Attempt atomic distance improvement
                    if new_dist < old_dist {
                        let mut current_dist = old_dist;

                        loop {
                            if new_dist >= current_dist {
                                break; // Another thread found better path
                            }

                            match distances[v].compare_exchange_weak(
                                current_dist,
                                new_dist,
                                Ordering::Release,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => {
                                    // Successfully updated distance
                                    predecessors[v].store(u, Ordering::Relaxed);

                                    // Categorize update for data structure
                                    if new_dist >= lower_bound && new_dist < upper_bound {
                                        local_updates.push((v, new_dist));
                                    } else if new_dist < lower_bound {
                                        local_batch_items.push((v, new_dist));
                                    }
                                    break;
                                }
                                Err(current) => {
                                    current_dist = current;
                                    // Retry with updated current value
                                }
                            }
                        }
                    }
                }

                // Merge local results
                updates.lock().unwrap().extend(local_updates);
                batch_prepend_items
                    .lock()
                    .unwrap()
                    .extend(local_batch_items);
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Apply updates to data structure (sequential for now)
        let updates_guard = updates.lock().unwrap();
        for (vertex, distance) in updates_guard.iter() {
            data_structure.insert(*vertex, *distance);
        }
        let batch_guard = batch_prepend_items.lock().unwrap();
        data_structure.batch_prepend(batch_guard.clone());
    }

    // Helper to clone the solver for threading
    fn clone_for_thread(&self) -> Self {
        ParallelSSSpSolver {
            graph: Arc::clone(&self.graph),
            distances: Arc::clone(&self.distances),
            predecessors: Arc::clone(&self.predecessors),
            complete: Arc::clone(&self.complete),
            k: self.k,
            t: self.t,
            num_threads: self.num_threads,
        }
    }

    // Placeholder for pivot extraction logic
    fn extract_pivots(&self, working_vec: &[usize], frontier: &[usize]) -> Vec<usize> {
        // This is a simplified placeholder. A real implementation would be more complex.
        if working_vec.len() > self.k * frontier.len() {
            frontier.to_vec()
        } else {
            working_vec.to_vec()
        }
    }
}

// --- Work-Stealing Solver ---

#[derive(Debug, Clone)]
enum WorkItem {
    ProcessPivot {
        level: usize,
        bound: f64,
        pivot: usize,
    },
    RelaxEdges {
        vertices: Vec<usize>,
        bounds: (f64, f64),
    },
    FindPivots {
        bound: f64,
        frontier: Vec<usize>,
    },
}

struct WorkStealingParallelSolver {
    graph: Arc<Graph>,
    distances: Arc<Vec<AtomicF64>>,
    global_queue: Arc<Injector<WorkItem>>,
    workers: Vec<Worker<WorkItem>>,
    stealers: Vec<Stealer<WorkItem>>,
    num_threads: usize,
}

impl WorkStealingParallelSolver {
    fn run_parallel(&self, source: usize) -> HashMap<usize, f64> {
        // Initialize with source vertex
        self.distances[source].store(0.0, Ordering::Relaxed);

        // Submit initial work
        let initial_work = WorkItem::ProcessPivot {
            level: self.max_level(),
            bound: INFINITY,
            pivot: source,
        };
        self.global_queue.push(initial_work);

        // Spawn worker threads
        let mut handles = Vec::new();
        for worker_id in 0..self.num_threads {
            // let solver = Arc::clone(self); // Needs to be Arc<Self>
            // let handle = thread::spawn(move || {
            //     solver.worker_loop(worker_id);
            // });
            // handles.push(handle);
        }

        // Wait for completion
        for handle in handles {
            handle.join().unwrap();
        }

        // Collect results
        self.collect_distances()
    }

    fn worker_loop(&self, worker_id: usize) {
        let worker = &self.workers[worker_id];

        loop {
            // Try to get work from local queue
            if let Some(work_item) = worker.pop() {
                self.process_work_item(work_item, worker_id);
                continue;
            }

            // Try to steal work from other workers or global queue
            if let Steal::Success(work_item) = self.global_queue.steal() {
                self.process_work_item(work_item, worker_id);
                continue;
            }

            let mut found_work = false;
            for stealer in &self.stealers {
                if let Steal::Success(work_item) = stealer.steal() {
                    self.process_work_item(work_item, worker_id);
                    found_work = true;
                    break;
                }
            }

            if !found_work {
                // No work available, check for termination condition
                if self.should_terminate() {
                    break;
                }
                std::thread::yield_now();
            }
        }
    }

    fn process_work_item(&self, _work_item: WorkItem, _worker_id: usize) {
        // Placeholder for processing logic
    }

    fn should_terminate(&self) -> bool {
        // Placeholder for termination logic
        true
    }

    fn max_level(&self) -> usize {
        // Placeholder
        0
    }

    fn collect_distances(&self) -> HashMap<usize, f64> {
        // Placeholder
        HashMap::new()
    }
}