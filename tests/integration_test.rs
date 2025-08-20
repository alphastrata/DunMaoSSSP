mod graph_loader;
use fast_sssp::SSSpSolver;
use graph_loader::read_dimacs_graph_for_fast_sssp;
use std::path::Path;

#[test]
fn sssp_from_file() {
    let graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));
    let mut solver = SSSpSolver::new(graph);
    // Find a path to a node we know is reachable
    let result = solver.solve(0, 3352);

    assert!(result.is_some());
    let (distance, path) = result.unwrap();
    assert!(distance > 0.0 && distance < f64::INFINITY);
    assert_eq!(path.first(), Some(&0));
    assert_eq!(path.last(), Some(&3352));
}
