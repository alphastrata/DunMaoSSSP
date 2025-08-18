mod graph_loader;
use fast_sssp::SSSpSolver;
use graph_loader::{read_dimacs_graph_for_fast_sssp, read_dimacs_graph_for_petgraph};
use petgraph::algo::dijkstra;
use std::path::Path;

#[test]
fn sanity_with_petgraph() {
    let (petgraph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
    let fast_graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));

    let source_node_id = 1;
    let goal_node_id = 3353;

    let source_node = node_map[&source_node_id];
    let goal_node = node_map[&goal_node_id];

    let petgraph_result = dijkstra(&petgraph, source_node, Some(goal_node), |e| *e.weight());

    let mut solver = SSSpSolver::new(fast_graph);
    let fast_sssp_result = solver.solve(source_node_id - 1, goal_node_id - 1);

    assert!(fast_sssp_result.is_some());
    let (fast_sssp_dist, _) = fast_sssp_result.unwrap();
    let petgraph_dist = petgraph_result.get(&goal_node).cloned().unwrap();

    assert!((fast_sssp_dist - petgraph_dist).abs() < 1e-9);
}