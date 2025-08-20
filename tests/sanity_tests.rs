mod graph_loader;

use crate::graph_loader::{read_dimacs_graph_for_fast_sssp, read_dimacs_graph_for_petgraph};
use fast_sssp::SSSpSolver;
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
fn sanity_with_petgraph() {
    let (petgraph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
    let fast_graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));

    let source_node_id = 1;
    let goal_node_id = 3353;

    let source_node = node_map[&source_node_id];
    let goal_node = node_map[&goal_node_id];

    let petgraph_result = dijkstra(&petgraph, source_node, Some(goal_node), |e| *e.weight());

    // Your fast SSSP result
    let mut fast_solver = SSSpSolver::new(fast_graph);
    let internal_source = convert_node_id_to_internal(source_node_id, &node_map);
    let internal_goal = convert_node_id_to_internal(goal_node_id, &node_map);

    let fast_sssp_result = fast_solver.solve(internal_source, internal_goal);

    assert!(fast_sssp_result.is_some());
    let (fast_sssp_dist, _) = fast_sssp_result.unwrap();
    let petgraph_dist = petgraph_result.get(&goal_node).cloned().unwrap();

    // Print the distances for debugging
    println!("fast_sssp distance: {}", fast_sssp_dist);
    println!("petgraph distance: {}", petgraph_dist);
    println!("Difference: {}", (fast_sssp_dist - petgraph_dist).abs());

    assert!((fast_sssp_dist - petgraph_dist).abs() < 1e-9);
}

#[test]
fn comprehensive_distance_comparison() {
    let (petgraph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
    let fast_graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));

    // Test multiple source/destination pairs
    let test_pairs = vec![
        (1, 100),
        (1, 500),
        (1, 1000),
        (1, 3353),
        (100, 200),
        (100, 1000),
        (100, 3353),
        (500, 1000),
        (500, 2000),
        (500, 3353),
    ];

    let mut fast_solver = SSSpSolver::new(fast_graph);
    let mut successful_tests = 0;
    let mut max_difference: f64 = 0.0;

    for (source_id, goal_id) in test_pairs {
        // Skip if nodes don't exist
        if !node_map.contains_key(&source_id) || !node_map.contains_key(&goal_id) {
            println!(
                "Skipping {}->{}: nodes not found in graph",
                source_id, goal_id
            );
            continue;
        }

        println!("Testing path from {} to {}", source_id, goal_id);

        // Petgraph computation
        let source_node = node_map[&source_id];
        let goal_node = node_map[&goal_id];
        let petgraph_result = dijkstra(&petgraph, source_node, Some(goal_node), |e| *e.weight());

        // Fast SSSP computation
        let internal_source = convert_node_id_to_internal(source_id, &node_map);
        let internal_goal = convert_node_id_to_internal(goal_id, &node_map);
        let fast_result = fast_solver.solve(internal_source, internal_goal);

        match (petgraph_result.get(&goal_node), fast_result) {
            (Some(&petgraph_dist), Some((fast_dist, _))) => {
                let diff: f64 = (fast_dist - petgraph_dist).abs();
                max_difference = max_difference.max(diff);

                println!(
                    "  ✓ Petgraph: {:.6}, Fast: {:.6}, Diff: {:.2e}",
                    petgraph_dist, fast_dist, diff
                );

                assert!(
                    diff < 1e-6,
                    "Distance mismatch for {}->{}: petgraph={}, fast={}, diff={}",
                    source_id,
                    goal_id,
                    petgraph_dist,
                    fast_dist,
                    diff
                );

                successful_tests += 1;
            }
            (None, None) => {
                println!("  ✓ Both algorithms agree: no path exists");
                successful_tests += 1;
            }
            (Some(petgraph_dist), None) => {
                panic!(
                    "❌ Petgraph found path (dist={:.6}), but fast algorithm didn't",
                    petgraph_dist
                );
            }
            (None, Some((fast_dist, _))) => {
                panic!(
                    "❌ Fast algorithm found path (dist={:.6}), but petgraph didn't",
                    fast_dist
                );
            }
        }
    }

    println!("\n📊 Summary:");
    println!("Successful tests: {}", successful_tests);
    println!("Maximum difference: {:.2e}", max_difference);
    assert!(
        successful_tests > 0,
        "No tests were successfully completed!"
    );
}

#[test]
fn validate_all_distances_from_source() {
    let (petgraph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
    let fast_graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));

    let source_node_id = 1;
    let source_node = node_map[&source_node_id];

    // Compute all distances with petgraph
    let petgraph_distances = dijkstra(&petgraph, source_node, None, |e| *e.weight());

    // Compute all distances with your algorithm
    let mut fast_solver = SSSpSolver::new(fast_graph);
    let internal_source = convert_node_id_to_internal(source_node_id, &node_map);
    let fast_distances = fast_solver.solve_all(internal_source);

    println!(
        "Petgraph found {} reachable nodes",
        petgraph_distances.len()
    );
    println!("Fast SSSP found {} reachable nodes", fast_distances.len());

    let mut compared_count = 0;
    let mut max_diff: f64 = 0.0;
    let mut total_diff: f64 = 0.0;

    // Compare distances for all reachable nodes
    for (&external_id, &petgraph_node) in &node_map {
        if let Some(&petgraph_dist) = petgraph_distances.get(&petgraph_node) {
            let internal_id = convert_node_id_to_internal(external_id, &node_map);

            if let Some(&fast_dist) = fast_distances.get(&internal_id) {
                let diff = (fast_dist - petgraph_dist).abs();
                max_diff = max_diff.max(diff);
                total_diff += diff;
                compared_count += 1;

                if diff > 1e-6 {
                    println!(
                        "❌ Large difference for node {}: petgraph={:.6}, fast={:.6}, diff={:.2e}",
                        external_id, petgraph_dist, fast_dist, diff
                    );
                }

                assert!(
                    diff < 1e-6,
                    "Distance mismatch for node {}: petgraph={:.6}, fast={:.6}, diff={:.2e}",
                    external_id,
                    petgraph_dist,
                    fast_dist,
                    diff
                );
            } else {
                panic!(
                    "❌ Petgraph found path to node {} (dist={:.6}), but fast algorithm didn't",
                    external_id, petgraph_dist
                );
            }
        }
    }

    // Check for extra nodes found by fast algorithm
    for (&internal_id, &fast_dist) in &fast_distances {
        let external_id = internal_id + 1; // Convert back to 1-based
        if let Some(&petgraph_node) = node_map.get(&external_id)
            && !petgraph_distances.contains_key(&petgraph_node)
        {
            panic!(
                "❌ Fast algorithm found path to node {} (dist={:.6}), but petgraph didn't",
                external_id, fast_dist
            );
        }
    }

    let avg_diff = if compared_count > 0 {
        total_diff / compared_count as f64
    } else {
        0.0
    };

    println!("\n📊 All-distances validation summary:");
    println!("Compared distances: {}", compared_count);
    println!("Maximum difference: {:.2e}", max_diff);
    println!("Average difference: {:.2e}", avg_diff);

    assert!(compared_count > 0, "No distances were compared!");
    println!("✅ All distance comparisons passed!");
}
