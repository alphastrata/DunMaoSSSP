#![allow(deprecated)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fast_sssp::graph::Graph;
use fast_sssp::sequential::SSSpSolver;
use fast_sssp::sequential_v2::DuanMaoSolverV2;
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::path::Path;

#[path = "../tests/graph_loader.rs"]
mod graph_loader;

fn run_v1(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = SSSpSolver::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn run_v2(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = DuanMaoSolverV2::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn benchmark(c: &mut Criterion) {
    let datasets = [
        ("Rome99", "tests/test_data/Rome99"),
        ("Wiki", "data/wiki-talk-graph.bin"),
    ];

    for (name, path) in &datasets {
        let graph = if name == &"Wiki" {
            Graph::from_file(Path::new(path)).unwrap()
        } else {
            graph_loader::read_dimacs_graph_for_fast_sssp(Path::new(path))
        };

        let mut rng = rand::rng();
        let mut nodes: Vec<usize> = (0..graph.vertices).collect();
        nodes.shuffle(&mut rng);
        let pairs: Vec<(usize, usize)> = nodes
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .take(10) // let us NOT be here all day..
            .collect();

        let mut group = c.benchmark_group(format!("{} SSSP (V1 vs V2)", name));

        group.bench_with_input(
            BenchmarkId::new("v1", name),
            &(&graph, &pairs),
            |b, (g, p)| b.iter(|| run_v1(black_box(g), black_box(p))),
        );

        group.bench_with_input(
            BenchmarkId::new("v2", name),
            &(&graph, &pairs),
            |b, (g, p)| b.iter(|| run_v2(black_box(g), black_box(p))),
        );

        group.finish();
    }
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
