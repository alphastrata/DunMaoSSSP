use crate::{DuanMaoSolverV2, Graph};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeIndexable, Visitable};
use std::collections::HashMap;
use std::hash::Hash;

/// Convert [`fast_sssp::Graph`] with [`fast_sssp::Graph::to_petgraph`] into a [`petgraph::Graph`], then you can use this like any of the algos in petgraph.
pub fn duan_mao<G>(graph: G, start: G::NodeId, goal: Option<G::NodeId>) -> HashMap<G::NodeId, f64>
where
    G: IntoEdges + Visitable + NodeIndexable + IntoNodeIdentifiers,
    G::NodeId: Eq + Hash + Clone,
    G::EdgeWeight: Into<f64> + Copy,
{
    let mut node_map = HashMap::new();
    let mut reverse_node_map = Vec::new();
    let mut our_graph = Graph::new(graph.node_bound());

    for node in graph.node_identifiers() {
        let idx = node_map.len();
        node_map.insert(node, idx);
        reverse_node_map.push(node);
    }

    for edge in graph.edge_references() {
        let source = node_map[&edge.source()];
        let target = node_map[&edge.target()];
        let weight: f64 = (*edge.weight()).into();
        our_graph.add_edge(source, target, weight);
    }

    let mut solver = DuanMaoSolverV2::new(our_graph.clone());
    let start_node_idx = node_map[&start];

    let mut result = HashMap::new();

    if let Some(goal_node) = goal {
        let goal_node_idx = node_map[&goal_node];
        if let Some((dist, _)) = solver.solve(start_node_idx, goal_node_idx) {
            result.insert(goal_node, dist);
        }
    } else {
        for (i, node_id) in reverse_node_map.iter().enumerate() {
            if let Some((dist, _)) = solver.solve(start_node_idx, i) {
                result.insert(*node_id, dist);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::Graph;

    #[test]
    fn duan_mao_with_petgraph() {
        let mut pg_graph = Graph::new();
        let a = pg_graph.add_node(()); // 0
        let b = pg_graph.add_node(()); // 1
        let c = pg_graph.add_node(()); // 2
        let d = pg_graph.add_node(()); // 3

        pg_graph.add_edge(a, b, 1.0);
        pg_graph.add_edge(a, c, 4.0);
        pg_graph.add_edge(b, c, 2.0);
        pg_graph.add_edge(b, d, 5.0);
        pg_graph.add_edge(c, d, 1.0);

        let result = duan_mao(&pg_graph, a, None);

        assert_eq!(result.get(&d), Some(&4.0));
    }

    #[test]
    fn duan_mao_with_goal() {
        let mut pg_graph = Graph::new();
        let a = pg_graph.add_node(()); // 0
        let b = pg_graph.add_node(()); // 1
        let c = pg_graph.add_node(()); // 2
        let d = pg_graph.add_node(()); // 3

        pg_graph.add_edge(a, b, 1.0);
        pg_graph.add_edge(a, c, 4.0);
        pg_graph.add_edge(b, c, 2.0);
        pg_graph.add_edge(b, d, 5.0);
        pg_graph.add_edge(c, d, 1.0);

        let result = duan_mao(&pg_graph, a, Some(d));

        assert_eq!(result.get(&d), Some(&4.0));
        assert_eq!(result.len(), 1);
    }
}
