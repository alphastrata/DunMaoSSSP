use criterion::{Criterion, criterion_group, criterion_main};
use fast_sssp::DuanMaoSolverV2;
use fast_sssp::graph::Graph;
use fast_sssp::parallel_v2::ParDuanMaoSolverV2;
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::path::Path;

#[path = "../tests/graph_loader.rs"]
mod graph_loader;

fn run_fast_sssp_duan_mao_v2(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = DuanMaoSolverV2::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn run_fast_sssp_parallel(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = ParDuanMaoSolverV2::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn benchmark_rome_parallel(c: &mut Criterion) {
    let path = Path::new("tests/test_data/Rome99");
    let fast_sssp_graph = graph_loader::read_dimacs_graph_for_fast_sssp(path);

    let mut rng = rand::rng();
    let mut nodes: Vec<usize> = (0..fast_sssp_graph.vertices).collect();
    nodes.shuffle(&mut rng);
    let pairs: Vec<(usize, usize)> = nodes
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .take(10)
        .collect();

    let mut group = c.benchmark_group("Rome99 SSSP (Parallel)");

    group.bench_function("fast_sssp_duan_mao_v2", |b| {
        b.iter(|| run_fast_sssp_duan_mao_v2(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.bench_function(format!("fast_sssp_parallel"), |b| {
        b.iter(|| run_fast_sssp_parallel(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.finish();
}

criterion_group!(benches, benchmark_rome_parallel);
criterion_main!(benches);
