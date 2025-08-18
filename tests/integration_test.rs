mod graph_loader;
use fast_sssp::SSSpSolver;
use graph_loader::read_dimacs_graph_for_fast_sssp;
use std::path::Path;

#[test]
fn sssp_from_file() {
    let graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));
    let mut solver = SSSpSolver::new(graph);
    let distances = solver.solve(0);

    assert_eq!(distances[0], 0.0);
    // We don't have the answer file, so we can't do a full comparison.
    // We'll just check that the solver runs and produces some finite distances.
    assert!(distances.iter().any(|&d| d > 0.0 && d.is_finite()));
}
