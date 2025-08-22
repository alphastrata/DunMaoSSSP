use crate::graph::Graph;
use crate::utils::{INFINITY, VertexDistance};
use crossbeam_utils::atomic::AtomicCell;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::sync::Mutex;
use std::sync::{Arc, MutexGuard};
use std::thread_local;

pub struct ParDuanMaoSolverV2 {
    pub graph: Graph,
    pub distances: Vec<AtomicCell<f64>>,
    pub predecessors: Vec<Mutex<Option<usize>>>,
    pub complete: Vec<AtomicCell<bool>>,
    pub k: usize,
    pub t: usize,
}

impl ParDuanMaoSolverV2 {
    pub fn new(graph: Graph) -> Self {
        let n = graph.vertices;
        let k = ((n as f64).ln().powf(1.0 / 3.0) * 2.0).floor() as usize;
        let t = ((n as f64).ln().powf(2.0 / 3.0)).floor() as usize;

        Self {
            distances: (0..n).map(|_| AtomicCell::new(INFINITY)).collect(),
            predecessors: (0..n).map(|_| Mutex::new(None)).collect(),
            complete: (0..n).map(|_| AtomicCell::new(false)).collect(),
            graph,
            k: k.max(3),
            t: t.max(2),
        }
    }

    pub fn solve(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        self.solve_duan_mao(source, goal)
    }

    fn solve_duan_mao(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        self.reset_state();
        self.distances[source].store(0.0);

        let max_level = ((self.graph.vertices as f64).ln() / self.t as f64).ceil() as usize;
        self.bmssp2(max_level, INFINITY, vec![source], Some(goal));

        let goal_dist = self.distances[goal].load();
        if goal_dist == INFINITY {
            None
        } else {
            Some((goal_dist, self.reconstruct_path(source, goal)))
        }
    }

    fn reset_state(&mut self) {
        for d in &self.distances {
            d.store(INFINITY);
        }
        for p in &self.predecessors {
            *p.lock().unwrap() = None;
        }
        for c in &self.complete {
            c.store(false);
        }
    }

    fn bmssp2(
        &mut self,
        level: usize,
        bound: f64,
        pivots: Vec<usize>,
        goal: Option<usize>,
    ) -> (f64, Vec<usize>) {
        if level == 0 {
            return self.base_case2(bound, pivots, goal);
        }

        if let Some(g) = goal
            && self.complete[g].load()
        {
            return (bound, Vec::new());
        }

        let (pivots, working_set) = self.find_pivots2(bound, &pivots);

        if working_set.len() > self.k * pivots.len() {
            return (bound, working_set);
        }

        let mut data_structure =
            EfficientDataStructure::new(2_usize.pow(((level - 1) * self.t).min(20) as u32), bound);

        for &pivot in &pivots {
            let dist = self.distances[pivot].load();
            if dist != INFINITY {
                data_structure.insert(pivot, dist);
            }
        }

        let mut result_set = Vec::new();
        let mut current_bound = pivots
            .iter()
            .filter(|&&v| self.distances[v].load() != INFINITY)
            .map(|&v| self.distances[v].load())
            .fold(INFINITY, f64::min);
        let max_result_size = self.k * 2_usize.pow((level * self.t).min(20) as u32);

        while result_set.len() < max_result_size && !data_structure.is_empty() {
            if let Some(g) = goal
                && self.complete[g].load()
            {
                break;
            }

            let (subset_bound, subset) = data_structure.pull();

            if subset.is_empty() {
                break;
            }

            let (sub_bound, sub_result) = self.bmssp2(level - 1, subset_bound, subset, goal);
            result_set.extend(&sub_result);

            self.edge_relaxation2_parallel(&sub_result, subset_bound, bound, &mut data_structure);
            current_bound = current_bound.min(sub_bound);
        }

        (current_bound, result_set)
    }

    fn base_case2(
        &mut self,
        bound: f64,
        frontier: Vec<usize>,
        goal: Option<usize>,
    ) -> (f64, Vec<usize>) {
        if frontier.is_empty() {
            return (bound, Vec::new());
        }

        let mut heap = BinaryHeap::new();
        for &start_node in &frontier {
            self.complete[start_node].store(true);
            let dist = self.distances[start_node].load();
            if dist < bound {
                heap.push(Reverse(VertexDistance::new(start_node, dist)));
            }
        }

        let mut result = Vec::new();
        let mut processed_count = 0;
        let limit = (self.k + frontier.len()).max(1000);

        while let Some(Reverse(VertexDistance { vertex, distance })) = heap.pop() {
            if distance > self.distances[vertex].load() {
                continue;
            }

            if let Some(g) = goal
                && vertex == g
            {
                result.push(vertex);
                break;
            }

            result.push(vertex);
            processed_count += 1;

            if processed_count > limit {
                break;
            }

            for edge in &self.graph.edges[vertex] {
                let new_dist = distance + edge.weight;
                let current_dist = self.distances[edge.to].load();
                if new_dist < current_dist && new_dist < bound {
                    self.distances[edge.to].store(new_dist);
                    *self.predecessors[edge.to].lock().unwrap() = Some(vertex);
                    heap.push(Reverse(VertexDistance::new(edge.to, new_dist)));
                }
            }
        }
        (bound, result)
    }

    fn reconstruct_path(&self, source: usize, goal: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = goal;
        while current != source {
            path.push(current);
            let pred = self.predecessors[current].lock().unwrap();
            if let Some(p) = *pred {
                current = p;
            } else {
                return Vec::new();
            }
        }
        path.push(source);
        path.reverse();
        path
    }

    fn find_pivots2(&mut self, bound: f64, frontier: &[usize]) -> (Vec<usize>, Vec<usize>) {
        // Parallel BFS exploration with work stealing
        let mut working_set: Vec<usize> = Vec::with_capacity(self.k * frontier.len() * 2);
        working_set.extend_from_slice(frontier);

        let explored = Arc::new(Mutex::new(HashSet::new()));
        for &node in frontier {
            explored.lock().unwrap().insert(node);
        }

        // Parallel exploration for k levels
        for level in 0..self.k {
            if working_set.is_empty() {
                break;
            }

            // Process current level in parallel
            let new_nodes: Vec<usize> = working_set
                .par_iter()
                .flat_map(|&u| {
                    let u_dist = self.distances[u].load();
                    let mut local_new = Vec::new();

                    for edge in &self.graph.edges[u] {
                        let v = edge.to;
                        let new_dist = u_dist + edge.weight;
                        let current_dist = self.distances[v].load();

                        if new_dist < current_dist && new_dist < bound {
                            // Use optimistic update without CAS
                            if new_dist < self.distances[v].fetch_min(new_dist) {
                                *self.predecessors[v].lock().unwrap() = Some(u);

                                let mut explored_lock = explored.lock().unwrap();
                                if explored_lock.insert(v) {
                                    local_new.push(v);
                                }
                            }
                        }
                    }
                    local_new
                })
                .collect();

            working_set = new_nodes;

            if working_set.len() > self.k * frontier.len() {
                break;
            }
        }

        let explored_set = explored.lock().unwrap().clone();
        (frontier.to_vec(), explored_set.into_iter().collect())
    }

    fn edge_relaxation2_parallel(
        &mut self,
        completed_vertices: &[usize],
        lower_bound: f64,
        upper_bound: f64,
        data_structure: &mut EfficientDataStructure,
    ) {
        // Batch processing with minimal synchronization
        let batches: Vec<(Vec<(usize, f64)>, Vec<(usize, f64)>)> = completed_vertices
            .par_chunks(rayon::current_num_threads() * 4)
            .map(|chunk| {
                let mut batch = Vec::new();
                let mut prepend_batch = Vec::new();

                for &u in chunk {
                    self.complete[u].store(true);
                    let u_dist = self.distances[u].load();

                    for edge in &self.graph.edges[u] {
                        let v = edge.to;
                        let new_dist = u_dist + edge.weight;
                        let current_dist = self.distances[v].load();

                        if new_dist < current_dist {
                            // Simple store - let the fastest thread win
                            self.distances[v].store(new_dist);
                            *self.predecessors[v].lock().unwrap() = Some(u);

                            if new_dist >= lower_bound && new_dist < upper_bound {
                                batch.push((v, new_dist));
                            } else if new_dist < lower_bound {
                                prepend_batch.push((v, new_dist));
                            }
                        }
                    }
                }

                // Return both batches as a tuple
                (batch, prepend_batch)
            })
            .collect();

        // Merge results
        let mut all_batches = Vec::new();
        let mut all_prepends = Vec::new();

        for (batch, prepend_batch) in batches {
            all_batches.extend(batch);
            all_prepends.extend(prepend_batch);
        }

        // Bulk insert
        for (v, dist) in all_batches {
            data_structure.insert(v, dist);
        }

        if !all_prepends.is_empty() {
            data_structure.batch_prepend(all_prepends);
        }
    }
}

trait AtomicF64Ext {
    fn fetch_min(&self, new_val: f64) -> f64;
}
impl AtomicF64Ext for AtomicCell<f64> {
    fn fetch_min(&self, new_val: f64) -> f64 {
        self.fetch_update(|current| {
            if new_val < current {
                Some(new_val)
            } else {
                None // Don't update
            }
        })
        .unwrap_or_else(|current| current)
    }
}

// EfficientDataStructure remains mostly the same, but optimized for parallel access
pub struct EfficientDataStructure {
    batch_blocks: Mutex<VecDeque<Vec<(usize, f64)>>>,
    sorted_blocks: Mutex<Vec<Vec<(usize, f64)>>>,
    block_size: usize,
    bound: f64,
}

impl EfficientDataStructure {
    pub fn new(block_size: usize, bound: f64) -> Self {
        Self {
            batch_blocks: Mutex::new(VecDeque::new()),
            sorted_blocks: Mutex::new(Vec::new()),
            block_size,
            bound,
        }
    }

    pub fn insert(&mut self, vertex: usize, distance: f64) {
        if distance < self.bound {
            let mut sorted_blocks = self.sorted_blocks.lock().unwrap();
            if sorted_blocks.is_empty() || sorted_blocks.last().unwrap().len() >= self.block_size {
                sorted_blocks.push(Vec::with_capacity(self.block_size));
            }
            sorted_blocks.last_mut().unwrap().push((vertex, distance));
        }
    }

    pub fn batch_prepend(&mut self, items: Vec<(usize, f64)>) {
        if !items.is_empty() {
            let mut batch_blocks = self.batch_blocks.lock().unwrap();
            batch_blocks.push_back(items);
        }
    }

    pub fn pull(&mut self) -> (f64, Vec<usize>) {
        // Try batch blocks first
        if let Some(mut block) = self.batch_blocks.lock().unwrap().pop_front() {
            block.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let vertices = block.into_iter().map(|(v, _)| v).collect();
            let min_dist = self.peek_min().unwrap_or(self.bound);
            return (min_dist, vertices);
        }

        // Then try sorted blocks
        if let Some(mut block) = self.sorted_blocks.lock().unwrap().pop() {
            block.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let vertices = block.into_iter().map(|(v, _)| v).collect();
            let min_dist = self.peek_min().unwrap_or(self.bound);
            return (min_dist, vertices);
        }

        (self.bound, Vec::new())
    }

    fn peek_min(&self) -> Option<f64> {
        let batch_blocks = self.batch_blocks.lock().unwrap();
        let sorted_blocks = self.sorted_blocks.lock().unwrap();

        let batch_min = batch_blocks
            .iter()
            .flat_map(|b| b.iter())
            .map(|(_, d)| *d)
            .fold(f64::INFINITY, f64::min);

        let sorted_min = sorted_blocks
            .iter()
            .flat_map(|b| b.iter())
            .map(|(_, d)| *d)
            .fold(f64::INFINITY, f64::min);

        let min = batch_min.min(sorted_min);
        if min == f64::INFINITY {
            None
        } else {
            Some(min)
        }
    }

    pub fn is_empty(&self) -> bool {
        let batch_blocks = self.batch_blocks.lock().unwrap();
        let sorted_blocks = self.sorted_blocks.lock().unwrap();
        batch_blocks.is_empty() && sorted_blocks.is_empty()
    }
}

#[test]
fn validate_parallel_correctness() {
    // Test specifically for parallel solver correctness on a smaller graph
    let mut graph = crate::graph::Graph::new(6);
    graph.add_edge(0, 1, 1.0);
    graph.add_edge(0, 2, 1.0);
    graph.add_edge(1, 3, 1.0);
    graph.add_edge(2, 4, 1.0);
    graph.add_edge(3, 5, 1.0);
    graph.add_edge(4, 5, 1.0);

    // Test all pairs
    let test_pairs = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 5),
        (2, 4),
        (2, 5),
        (3, 5),
        (4, 5),
    ];

    for (source, goal) in test_pairs {
        println!("Testing path from {} to {}", source, goal);

        // Sequential solver
        let mut sequential_solver = crate::DuanMaoSolverV2::new(graph.clone());
        let sequential_result = sequential_solver.solve(source, goal).unwrap();

        // Parallel solver
        let mut parallel_solver = ParDuanMaoSolverV2::new(graph.clone());
        let parallel_result = parallel_solver.solve(source, goal).unwrap();

        assert_eq!(sequential_result.0, parallel_result.0);
        assert_eq!(sequential_result.1.len(), parallel_result.1.len());
    }
}
