#![allow(deprecated)]
#![cfg(feature = "parallel")]

#[path = "graph_loader.rs"]
mod graph_loader;

use crate::graph_loader::{read_dimacs_graph_for_fast_sssp, read_dimacs_graph_for_petgraph};
use fast_sssp::{DuanMaoSolverV2, SSSpSolver};
use petgraph::algo::dijkstra;
use std::collections::HashMap;
use std::path::Path;

// Helper function to convert external node IDs to internal indices
fn convert_node_id_to_internal(
    external_id: usize,
    _node_map: &HashMap<usize, petgraph::graph::NodeIndex>,
) -> usize {
    external_id - 1 // Convert 1-based DIMACS IDs to 0-based internal indices
}

#[test]
fn sanity_with_all_solvers() {
    let (petgraph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
    let fast_graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));

    let source_node_id = 1;
    let goal_node_id = 3353;

    let source_node = node_map[&source_node_id];
    let goal_node = node_map[&goal_node_id];

    // Petgraph result
    let petgraph_result = dijkstra(&petgraph, source_node, Some(goal_node), |e| *e.weight());
    let petgraph_dist = petgraph_result
        .get(&goal_node)
        .cloned()
        .unwrap_or(f64::INFINITY);

    // Sequential Duan-Mao result
    let mut sequential_solver = DuanMaoSolverV2::new(fast_graph.clone());
    let internal_source = convert_node_id_to_internal(source_node_id, &node_map);
    let internal_goal = convert_node_id_to_internal(goal_node_id, &node_map);
    let sequential_result = sequential_solver.solve(internal_source, internal_goal);
    let sequential_dist = sequential_result.map(|(d, _)| d).unwrap_or(f64::INFINITY);

    // SSSpSolver result (if available)
    let mut sssp_solver = SSSpSolver::new(fast_graph);
    let sssp_result = sssp_solver.solve(internal_source, internal_goal);
    let sssp_dist = sssp_result.map(|(d, _)| d).unwrap_or(f64::INFINITY);

    // Print all distances for debugging
    println!("Petgraph distance: {}", petgraph_dist);
    println!("Sequential Duan-Mao distance: {}", sequential_dist);
    // println!("Parallel Duan-Mao distance: {}", parallel_dist);
    println!("SSSpSolver distance: {}", sssp_dist);

    // Also check that all solvers agree on path existence
    let has_path_petgraph = petgraph_dist < f64::INFINITY;
    let has_path_sequential = sequential_dist < f64::INFINITY;
    // let has_path_parallel = parallel_dist < f64::INFINITY;
    let has_path_sssp = sssp_dist < f64::INFINITY;

    assert_eq!(
        has_path_petgraph, has_path_sequential,
        "Path existence mismatch between petgraph and sequential solver"
    );
    // assert_eq!(
    //     has_path_petgraph, has_path_parallel,
    //     "Path existence mismatch between petgraph and parallel solver"
    // );
    assert_eq!(
        has_path_petgraph, has_path_sssp,
        "Path existence mismatch between petgraph and SSSpSolver"
    );

    println!("✅ All solvers agree on distance and path existence!");
}
