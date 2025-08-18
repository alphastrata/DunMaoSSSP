use std::time::Instant;
use std::{hint::black_box, path::Path};

use fast_sssp::{Graph, SSSpSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/wiki-talk-graph.bin");

    if !data_path.exists() {
        println!("Wiki-Talk graph not found!");
        println!("Please run: cargo run --bin fetch_data");
        return Ok(())
    }

    println!("Loading Wiki-Talk dataset...");
    let graph = Graph::from_file(data_path)?;

    println!(
        "Graph loaded: {} vertices, {} edges",
        graph.vertices,
        graph.edge_count()
    );

    println!(
        "\nNote: This is a real-world directed graph with {} vertices and {} edges",
        graph.vertices,
        graph.edge_count()
    );

    println!("\nBenchmarking on Wiki-Talk dataset:");
    /*
        2_394_385 vertices, 5_021_410 edges
    */

    println!(
        "{:<8} {:<8} {:<15} {:<15} {:<12.2}",
        "Source", "Goal", "Dijkstra (ms)", "New Algo (ms)", "Speedup"
    );
    println!("{}", "-".repeat(65));

    // Test on a subset of source vertices
    let test_pairs = vec![(0, 20000), (100, 30000), (1000, 40000), (5000, 50000)];

    for &(source, goal) in &test_pairs {
        if source >= graph.vertices || goal >= graph.vertices {
            continue;
        }

        // Benchmark Dijkstra
        let mut solver1 = SSSpSolver::new(graph.clone());
        let start = Instant::now();
        let result1 = solver1.dijkstra(source, Some(goal));
        let dijkstra_time = start.elapsed().as_millis();

        // VS the new
        let mut solver2 = SSSpSolver::new(graph.clone());
        let start = Instant::now();
        let result2 = solver2.solve(source, goal);
        let new_algo_time = start.elapsed().as_millis();

        black_box((result1, result2)); // JIC rustc tries to be too clever.

        let speedup = if new_algo_time > 0 {
            dijkstra_time as f64 / new_algo_time as f64
        } else {
            0.0
        };

        println!(
            "{:<8} {:<8} {:<15} {:<15} {:<12.2}x",
            source, goal, dijkstra_time, new_algo_time, speedup
        );
    }

    Ok(())
}