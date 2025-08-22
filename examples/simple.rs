fn main() {
    // Create a larger graph to showcase the new algorithm
    let mut graph = fast_sssp::Graph::new(25);

    // Create a more complex graph structure
    for i in 0..24 {
        graph.add_edge(i, i + 1, (i % 5 + 1) as f64);
        if i >= 5 {
            graph.add_edge(i, i - 5, 2.0);
        }
    }

    // Add some cross connections
    graph.add_edge(0, 10, 15.0);
    graph.add_edge(5, 20, 12.0);
    graph.add_edge(12, 3, 8.0);
    graph.add_edge(18, 7, 6.0);

    let mut solver = fast_sssp::DuanMaoSolverV2::new(graph);
    let distances = solver.solve(0, 18);

    println!("Shortest distances from vertex 0:");
}
