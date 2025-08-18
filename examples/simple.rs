use fast_sssp::{Graph, SSSpSolver};

fn main() {
    // Create a simple graph
    let mut graph = Graph::new(6);
    graph.add_edge(0, 1, 7.0);
    graph.add_edge(0, 2, 9.0);
    graph.add_edge(0, 5, 14.0);
    graph.add_edge(1, 2, 10.0);
    graph.add_edge(1, 3, 15.0);
    graph.add_edge(2, 3, 11.0);
    graph.add_edge(2, 5, 2.0);
    graph.add_edge(3, 4, 6.0);
    graph.add_edge(4, 5, 9.0);

    let mut solver = SSSpSolver::new(graph);
    let source = 0;
    let goal = 4;

    println!(
        "Finding shortest path from vertex {} to {}...",
        source, goal
    );

    if let Some((distance, path)) = solver.solve(source, goal) {
        println!("Shortest distance: {:.1}", distance);
        println!("Path: {:?}", path);
    } else {
        println!("No path found from {} to {}", source, goal);
    }
}