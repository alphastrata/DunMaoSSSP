use crate::graph::Graph;
use crate::utils::{INFINITY, VertexDistance};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

pub struct DuanMaoSolverV2 {
    graph: Graph,
    distances: Vec<f64>,
    predecessors: Vec<Option<usize>>,
    complete: Vec<bool>,
    k: usize,
    t: usize,
}

impl DuanMaoSolverV2 {
    pub fn new(graph: Graph) -> Self {
        let n = graph.vertices;
        let k = ((n as f64).ln().powf(1.0 / 3.0)).floor() as usize;
        let t = ((n as f64).ln().powf(2.0 / 3.0)).floor() as usize;

        Self {
            distances: vec![INFINITY; n],
            predecessors: vec![None; n],
            complete: vec![false; n],
            graph,
            k: k.max(3),
            t: t.max(2),
        }
    }

    pub fn solve(&mut self, source: usize, goal: usize) -> Option<(f64, Vec<usize>)> {
        self.reset_state();
        self.distances[source] = 0.0;

        let max_level = ((self.graph.vertices as f64).ln() / self.t as f64).ceil() as usize;
        self.bmssp2(max_level, INFINITY, vec![source], Some(goal));

        if self.distances[goal] == INFINITY {
            None
        } else {
            Some((self.distances[goal], self.reconstruct_path(source, goal)))
        }
    }

    fn reset_state(&mut self) {
        self.distances.fill(INFINITY);
        self.predecessors.fill(None);
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
            && self.complete[g] {
                return (bound, Vec::new());
            }

        let (pivots, working_set) = self.find_pivots2(bound, &pivots);

        if working_set.len() > self.k * pivots.len() {
            return (bound, working_set);
        }

        let mut data_structure =
            AdaptiveDataStructure2::new(2_usize.pow(((level - 1) * self.t).min(20) as u32), bound);

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

        while result_set.len() < max_result_size && !data_structure.is_empty() {
            if let Some(g) = goal
                && self.complete[g] {
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
            if self.distances[start_node] < bound {
                heap.push(Reverse(VertexDistance::new(
                    start_node,
                    self.distances[start_node],
                )));
            }
        }

        let mut result = Vec::new();
        let mut processed_count = 0;
        let limit = self.k * frontier.len();

        while let Some(Reverse(VertexDistance { vertex, distance })) = heap.pop() {
            if self.complete[vertex] || distance > self.distances[vertex] {
                continue;
            }

            if let Some(g) = goal
                && vertex == g {
                    self.complete[vertex] = true;
                    result.push(vertex);
                    break;
                }

            self.complete[vertex] = true;
            result.push(vertex);
            processed_count += 1;

            if processed_count > limit {
                break;
            }

            for edge in &self.graph.edges[vertex] {
                let new_dist = distance + edge.weight;
                if new_dist < self.distances[edge.to] && new_dist < bound {
                    self.distances[edge.to] = new_dist;
                    self.predecessors[edge.to] = Some(vertex);
                    heap.push(Reverse(VertexDistance::new(edge.to, new_dist)));
                }
            }
        }
        (bound, result)
    }

    fn find_pivots2(&mut self, bound: f64, frontier: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut working_set: HashSet<usize> = frontier.iter().copied().collect();
        let mut current_layer: Vec<usize> = frontier.to_vec();

        for _ in 0..self.k {
            let mut next_layer = HashSet::new();
            for &u in &current_layer {
                for edge in &self.graph.edges[u] {
                    let v = edge.to;
                    let new_dist = self.distances[u] + edge.weight;
                    if new_dist < self.distances[v] && new_dist < bound {
                        self.distances[v] = new_dist;
                        self.predecessors[v] = Some(u);
                        if working_set.insert(v) {
                            next_layer.insert(v);
                        }
                    }
                }
            }

            if next_layer.is_empty() {
                break;
            }
            current_layer = next_layer.into_iter().collect();

            if working_set.len() > self.k * frontier.len() {
                return (frontier.to_vec(), working_set.into_iter().collect());
            }
        }

        (frontier.to_vec(), working_set.into_iter().collect())
    }

    fn edge_relaxation2(
        &mut self,
        completed_vertices: &[usize],
        lower_bound: f64,
        upper_bound: f64,
        data_structure: &mut AdaptiveDataStructure2,
    ) {
        let mut batch_prepend_list = Vec::new();
        for &u in completed_vertices {
            self.complete[u] = true;
            for edge in &self.graph.edges[u] {
                let v = edge.to;
                let new_dist = self.distances[u] + edge.weight;
                if new_dist < self.distances[v] {
                    self.distances[v] = new_dist;
                    self.predecessors[v] = Some(u);
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
            if let Some(pred) = self.predecessors[current] {
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

pub struct AdaptiveDataStructure2 {
    data: BinaryHeap<Reverse<VertexDistance>>,
    capacity: usize,
    bound: f64,
}

impl AdaptiveDataStructure2 {
    pub fn new(capacity: usize, bound: f64) -> Self {
        Self {
            data: BinaryHeap::new(),
            capacity,
            bound,
        }
    }

    pub fn insert(&mut self, vertex: usize, distance: f64) {
        if distance < self.bound {
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
