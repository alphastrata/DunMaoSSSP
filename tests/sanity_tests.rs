mod graph_loader;
use fast_sssp::SSSpSolver;
use graph_loader::{read_dimacs_graph_for_fast_sssp, read_dimacs_graph_for_petgraph};
use petgraph::algo::dijkstra;
use std::path::Path;

#[test]
fn sanity_with_petgraph() {
    let (petgraph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
    let fast_graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));

    let source_node = node_map[&1];
    let petgraph_distances = dijkstra(&petgraph, source_node, None, |e| *e.weight());

    let mut solver = SSSpSolver::new(fast_graph);
    let fast_sssp_distances = solver.solve(0);

    for (node_id, node_index) in &node_map {
        let petgraph_dist = petgraph_distances
            .get(node_index)
            .cloned()
            .unwrap_or(f64::INFINITY);
        let fast_sssp_dist = fast_sssp_distances[*node_id - 1];

        if petgraph_dist.is_finite() && fast_sssp_dist.is_finite() {
            assert!(
                (petgraph_dist - fast_sssp_dist).abs() < 1e-9,
                "Mismatch at node {}: petgraph={}, fast_sssp={}",
                node_id,
                petgraph_dist,
                fast_sssp_dist
            );
        } else {
            assert_eq!(
                petgraph_dist.is_finite(),
                fast_sssp_dist.is_finite(),
                "Mismatch in reachability at node {}",
                node_id
            );
        }
    }
}
