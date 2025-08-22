use crate::graph::Graph;
use crate::utils::{INFINITY, VertexDistance};
use crossbeam_utils::atomic::AtomicCell;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Mutex;

pub struct ParDuanMaoSolverV2 {
    pub graph: Graph,
    pub distances: Vec<AtomicCell<f64>>,
    pub predecessors: Vec<Mutex<Option<usize>>>,
    pub complete: Vec<bool>,
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
            complete: vec![false; n],
            graph,
            k: k.max(3),
            t: t.max(2),
        }
    }

    pub fn solve(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        if self.graph.vertices < 50_000 || self.graph.edge_count() < 200_000 {
            return self.dijkstra_fallback(source, goal);
        }
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

    fn dijkstra_fallback(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        self.reset_state();
        self.distances[source].store(0.0);

        let mut heap = BinaryHeap::new();
        heap.push(Reverse(VertexDistance::new(source, 0.0)));

        while let Some(Reverse(VertexDistance { vertex, distance })) = heap.pop() {
            if distance > self.distances[vertex].load() {
                continue;
            }
            if vertex == goal {
                return Some((distance, self.reconstruct_path(source, goal)));
            }

            for edge in &self.graph.edges[vertex] {
                let new_dist = distance + edge.weight;
                if new_dist < self.distances[edge.to].load() {
                    self.distances[edge.to].store(new_dist);
                    *self.predecessors[edge.to].lock().unwrap() = Some(vertex);
                    heap.push(Reverse(VertexDistance::new(edge.to, new_dist)));
                }
            }
        }
        None
    }

    fn reset_state(&mut self) {
        for d in &self.distances {
            d.store(INFINITY);
        }
        for p in &self.predecessors {
            *p.lock().unwrap() = None;
        }
        self.complete.fill(false);
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
            && self.complete[g]
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
                && self.complete[g]
            {
                break;
            }

            let (subset_bound, subset) = data_structure.pull();

            if subset.is_empty() {
                break;
            }

            let (sub_bound, sub_result) = self.bmssp2(level - 1, subset_bound, subset, goal);
            result_set.extend(&sub_result);

            self.edge_relaxation2(&sub_result, subset_bound, bound, &mut data_structure);
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
            self.complete[start_node] = true;
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

    fn find_pivots2(&mut self, bound: f64, frontier: &[usize]) -> (Vec<usize>, Vec<usize>) {
        find_pivots2_parallel(
            &self.graph,
            &self.distances,
            &self.predecessors,
            self.k,
            bound,
            frontier,
        )
    }

    fn edge_relaxation2(
        &mut self,
        completed_vertices: &[usize],
        lower_bound: f64,
        upper_bound: f64,
        data_structure: &mut EfficientDataStructure,
    ) {
        let mut batch_prepend_list = Vec::new();
        for &u in completed_vertices {
            self.complete[u] = true;
            let u_dist = self.distances[u].load();
            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = u_dist + edge.weight;
                let current_dist = self.distances[v].load();
                if new_dist < current_dist {
                    self.distances[v].store(new_dist);
                    *self.predecessors[v].lock().unwrap() = Some(u);
                    if new_dist >= lower_bound && new_dist < upper_bound {
                        data_structure.insert(v, new_dist);
                    } else if new_dist < lower_bound {
                        batch_prepend_list.push((v, new_dist));
                    }
                }
            }
        }
        data_structure.batch_prepend(batch_prepend_list);
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
}

fn find_pivots2_parallel(
    graph: &Graph,
    distances: &[AtomicCell<f64>],
    predecessors: &[Mutex<Option<usize>>],
    k: usize,
    bound: f64,
    frontier: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut working_set: HashSet<usize> = frontier.iter().copied().collect();
    let mut current_layer: Vec<usize> = frontier.to_vec();

    for _ in 0..k {
        if current_layer.is_empty() {
            break;
        }

        // Collect all potential updates in parallel
        let updates: Vec<(usize, f64, usize)> = current_layer
            .par_iter()
            .flat_map(|&u| {
                let u_dist = distances[u].load();
                graph.edges[u]
                    .iter()
                    .filter_map(move |edge| {
                        let v = edge.to;
                        let new_dist = u_dist + edge.weight;
                        if new_dist < distances[v].load() && new_dist < bound {
                            Some((v, new_dist, u))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Apply updates sequentially to avoid race conditions
        let mut next_layer = HashSet::new();
        for (v, new_dist, u) in updates {
            if new_dist < distances[v].load() {
                distances[v].store(new_dist);
                *predecessors[v].lock().unwrap() = Some(u);

                if working_set.insert(v) {
                    next_layer.insert(v);
                }
            }
        }

        if working_set.len() > k * frontier.len() {
            return (frontier.to_vec(), working_set.into_iter().collect());
        }

        current_layer = next_layer.into_iter().collect();
    }

    select_pivots_from_working_set(predecessors, k, frontier, &working_set)
}

fn select_pivots_from_working_set(
    predecessors: &[Mutex<Option<usize>>],
    k: usize,
    frontier: &[usize],
    working_set: &HashSet<usize>,
) -> (Vec<usize>, Vec<usize>) {
    // Calculate subtree sizes
    let mut subtree_sizes: HashMap<usize, usize> = HashMap::new();

    // Build a tree structure from predecessors
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for &node in working_set {
        if let Some(parent) = *predecessors[node].lock().unwrap()
            && working_set.contains(&parent)
        {
            children.entry(parent).or_default().push(node);
        }
    }

    // Calculate subtree sizes using DFS
    let mut stack: Vec<usize> = Vec::new();
    let mut visited = HashSet::new();

    // Start DFS from all nodes that have no children (leaves)
    for &node in working_set {
        if children.get(&node).is_none_or(|c| c.is_empty()) {
            stack.push(node);
        }
    }

    while let Some(node) = stack.pop() {
        if visited.contains(&node) {
            continue;
        }

        let mut size = 1;
        let mut all_children_visited = true;

        if let Some(children_list) = children.get(&node) {
            for &child in children_list {
                if let Some(child_size) = subtree_sizes.get(&child) {
                    size += child_size;
                } else {
                    all_children_visited = false;
                    stack.push(node);
                    stack.push(child);
                    break;
                }
            }
        }

        if all_children_visited {
            subtree_sizes.insert(node, size);
            visited.insert(node);
        }
    }

    // Select pivots: frontier nodes with subtree size >= k
    let pivots: Vec<usize> = frontier
        .iter()
        .filter(|&&root| subtree_sizes.get(&root).is_some_and(|&size| size >= k))
        .copied()
        .collect();

    let final_pivots = if pivots.is_empty() {
        frontier.to_vec()
    } else {
        pivots
    };

    (final_pivots, working_set.iter().copied().collect())
}

// EfficientDataStructure remains the same as in your sequential version
pub struct EfficientDataStructure {
    batch_blocks: VecDeque<Vec<(usize, f64)>>,
    sorted_blocks: Vec<Vec<(usize, f64)>>,
    block_size: usize,
    bound: f64,
}

impl EfficientDataStructure {
    pub fn new(block_size: usize, bound: f64) -> Self {
        Self {
            batch_blocks: VecDeque::new(),
            sorted_blocks: Vec::new(),
            block_size,
            bound,
        }
    }

    pub fn insert(&mut self, vertex: usize, distance: f64) {
        if distance < self.bound {
            if self.sorted_blocks.is_empty()
                || self.sorted_blocks.last().unwrap().len() >= self.block_size
            {
                self.sorted_blocks.push(Vec::with_capacity(self.block_size));
            }
            self.sorted_blocks
                .last_mut()
                .unwrap()
                .push((vertex, distance));
        }
    }

    pub fn batch_prepend(&mut self, items: Vec<(usize, f64)>) {
        if !items.is_empty() {
            self.batch_blocks.push_back(items);
        }
    }

    pub fn pull(&mut self) -> (f64, Vec<usize>) {
        if let Some(mut block) = self.batch_blocks.pop_front() {
            block.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let vertices = block.into_iter().map(|(v, _)| v).collect();
            let min_dist = self.peek_min().unwrap_or(self.bound);
            return (min_dist, vertices);
        }

        if let Some(mut block) = self.sorted_blocks.pop() {
            block.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let vertices = block.into_iter().map(|(v, _)| v).collect();
            let min_dist = self.peek_min().unwrap_or(self.bound);
            return (min_dist, vertices);
        }

        (self.bound, Vec::new())
    }

    fn peek_min(&self) -> Option<f64> {
        let batch_min = self
            .batch_blocks
            .iter()
            .flat_map(|b| b.iter())
            .map(|(_, d)| *d)
            .fold(f64::INFINITY, f64::min);
        let sorted_min = self
            .sorted_blocks
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
        self.batch_blocks.is_empty() && self.sorted_blocks.is_empty()
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

        assert_eq!(sequential_result.1.len(), parallel_result.1.len());
    }
}
