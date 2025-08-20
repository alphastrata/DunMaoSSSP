#[cfg(feature = "parallel")]
use criterion::{Criterion, criterion_group, criterion_main};
use fast_sssp::graph::Graph;
use fast_sssp::parallel::ParallelSSSpSolver;
use fast_sssp::sequential::SSSpSolver;
use petgraph::algo::dijkstra;
use petgraph::graph::DiGraph;
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::path::Path;

#[path = "../tests/graph_loader.rs"]
mod graph_loader;

fn run_fast_sssp_sequential(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = SSSpSolver::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn run_fast_sssp_parallel(graph: &Graph, pairs: &[(usize, usize)], num_threads: usize) {
    let solver = ParallelSSSpSolver::new(graph.clone(), num_threads);
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

#[allow(dead_code)]
fn run_petgraph_dijkstra(graph: &DiGraph<(), f64>, pairs: &[(usize, usize)]) {
    for (source, goal) in pairs {
        let source_node = graph.node_indices().find(|i| i.index() == *source).unwrap();
        let goal_node = graph.node_indices().find(|i| i.index() == *goal).unwrap();
        black_box(dijkstra(&graph, source_node, Some(goal_node), |e| {
            *e.weight()
        }));
    }
}

fn benchmark(c: &mut Criterion) {
    let path = Path::new("data/wiki-talk-graph.bin");
    let fast_sssp_graph = Graph::from_file(path).unwrap();

    let mut rng = rand::rng();
    let mut nodes: Vec<usize> = (0..fast_sssp_graph.vertices).collect();
    nodes.shuffle(&mut rng);
    let pairs: Vec<(usize, usize)> = nodes
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .take(100)
        .collect();

    let mut group = c.benchmark_group("WikiTalk SSSP");

    group.bench_function("fast_sssp_sequential", |b| {
        b.iter(|| run_fast_sssp_sequential(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    for threads in [8, 12, 16, 24, 32].iter() {
        group.bench_function(format!("fast_sssp_parallel_{}_threads", threads), |b| {
            b.iter(|| {
                run_fast_sssp_parallel(black_box(&fast_sssp_graph), black_box(&pairs), *threads)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
