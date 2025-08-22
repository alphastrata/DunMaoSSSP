use crate::graph::Graph;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::collections::{HashMap, HashSet};

pub fn find_pivots2_parallel(
    graph: &Graph,
    distances: &mut Vec<f64>,
    predecessors: &mut Vec<Option<usize>>,
    k: usize,
    bound: f64,
    frontier: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut working_set: HashSet<usize> = HashSet::with_capacity(k * frontier.len());
    working_set.extend(frontier.iter().copied());
    let mut current_layer: Vec<usize> = frontier.to_vec();

    for _ in 0..k {
        if current_layer.is_empty() {
            break;
        }

        let chunk_size = (current_layer.len() / rayon::current_num_threads()).max(1);
        let all_updates: Vec<(usize, f64, usize)> = current_layer
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                let mut local_updates = Vec::new();

                for &u in chunk {
                    let u_dist = distances[u]; // Read-only, no sync needed
                    for edge in &graph.edges[u] {
                        let v = edge.to;
                        let new_dist = u_dist + edge.weight;

                        if new_dist < distances[v] && new_dist < bound {
                            local_updates.push((v, new_dist, u));
                        }
                    }
                }
                local_updates
            })
            .collect();

        let mut next_layer = HashSet::new();
        for (v, new_dist, u) in all_updates {
            if new_dist < distances[v] {
                distances[v] = new_dist;
                predecessors[v] = Some(u);

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
    predecessors: &mut Vec<Option<usize>>,
    k: usize,
    frontier: &[usize],
    working_set: &HashSet<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let subtree_sizes = predecessors
        .iter()
        .filter_map(|p| *p)
        .filter(|p| working_set.contains(p))
        .fold(HashMap::new(), |mut acc, p| {
            *acc.entry(p).or_insert(0) += 1;
            acc
        });

    let pivots: Vec<usize> = subtree_sizes
        .par_iter()
        .filter(|&(_, size)| *size >= k)
        .filter_map(|(&root, _)| {
            if frontier.contains(&root) {
                Some(root)
            } else {
                None
            }
        })
        .collect();

    let final_pivots = if pivots.is_empty() {
        frontier.to_vec()
    } else {
        pivots
    };

    (final_pivots, working_set.clone().into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::utils::INFINITY;

    #[test]
    fn test_find_pivots2_parallel() {
        let mut graph = Graph::new(6);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(0, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 4, 1.0);
        graph.add_edge(3, 5, 1.0);
        graph.add_edge(4, 5, 1.0);

        let mut distances = vec![INFINITY; graph.vertices];
        let mut predecessors = vec![None; graph.vertices];
        distances[0] = 0.0;

        let (pivots, working_set) =
            find_pivots2_parallel(&graph, &mut distances, &mut predecessors, 1, INFINITY, &[0]);

        assert_eq!(pivots, vec![0]);
        assert_eq!(working_set.len(), 6);
    }
}
