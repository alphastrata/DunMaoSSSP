use std::time::Instant;
use std::{hint::black_box, path::Path};

use fast_sssp::{Graph, SSSpSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/wiki-talk-graph.bin");

    if !data_path.exists() {
        println!("Wiki-Talk graph not found!");
        println!("Please run: cargo run --bin fetch_data");
        return Ok(());
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
        "{:<8} {:<15} {:<15} {:<12.2}",
        "Source", "Dijkstra (ms)", "New Algo (ms)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    // Reuse the same graphs for all sssp measures.
    let mut solver1 = SSSpSolver::new(graph.clone());
    let mut solver2 = SSSpSolver::new(graph.clone());

    // Test on a subset of source vertices
    let test_sources = vec![0, 100, 1000, 5000];

    for &source in &test_sources {
        if source >= graph.vertices {
            continue;
        }

        // Benchmark Dijkstra
        let start = Instant::now();
        let distances1 = solver1.dijkstra(source);
        let dijkstra_time = start.elapsed().as_millis();

        // VS the new
        let start = Instant::now();
        let distances2 = solver2.solve(source);
        let new_algo_time = start.elapsed().as_millis();

        assert_eq!(distances2.len(), distances1.len()); // Should be the same.
        black_box((_distances1, _distances2)); // JIC rustc tries to be too clever.

        let speedup = if new_algo_time > 0 {
            dijkstra_time as f64 / new_algo_time as f64
        } else {
            0.0
        };

        println!(
            "{:<8} {:<15} {:<15} {:<12.2}x",
            source, dijkstra_time, new_algo_time, speedup
        );
    }

    Ok(())
}
