use fast_sssp::{Graph as FastGraph, SSSpSolver};
use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn read_dimacs_graph_for_petgraph(path: &Path) -> (DiGraph<(), f64>, HashMap<usize, NodeIndex>) {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();
    let mut node_map = HashMap::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "c" => continue, // Comment
            "p" => {
                // Problem line: p sp <nodes> <edges>
                let vertices = parts[2].parse::<usize>().unwrap();
                for i in 1..=vertices {
                    let node = graph.add_node(());
                    node_map.insert(i, node);
                }
            }
            "a" => {
                // Arc descriptor: a <from> <to> <weight>
                let from = parts[1].parse::<usize>().unwrap();
                let to = parts[2].parse::<usize>().unwrap();
                let weight = parts[3].parse::<f64>().unwrap();
                let from_node = node_map[&from];
                let to_node = node_map[&to];
                graph.add_edge(from_node, to_node, weight);
            }
            _ => continue,
        }
    }
    (graph, node_map)
}

fn read_dimacs_graph_for_fast_sssp(path: &Path) -> FastGraph {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut graph = None;

    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "c" => continue, // Comment
            "p" => {
                // Problem line: p sp <nodes> <edges>
                let vertices = parts[2].parse::<usize>().unwrap();
                graph = Some(FastGraph::new(vertices));
            }
            "a" => {
                // Arc descriptor: a <from> <to> <weight>
                let from = parts[1].parse::<usize>().unwrap() - 1; // Adjust for 0-based indexing
                let to = parts[2].parse::<usize>().unwrap() - 1; // Adjust for 0-based indexing
                let weight = parts[3].parse::<f64>().unwrap();
                if let Some(g) = &mut graph {
                    g.add_edge(from, to, weight);
                }
            }
            _ => continue,
        }
    }
    graph.unwrap()
}

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
