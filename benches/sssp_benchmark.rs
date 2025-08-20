// use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
// use fast_sssp::{dun_mao_all_distances, Graph, SSSpSolver};
// use rand::rngs::StdRng;
// use rand::{Rng, SeedableRng};

// fn create_sparse_graph(n: usize, density: f64, seed: u64) -> Graph {
//     let mut rng = StdRng::seed_from_u64(seed);
//     let mut graph = Graph::new(n);

//     let m = ((n as f64) * density).round() as usize;

//     // Ensure connectivity by creating a spanning tree first
//     for i in 1..n {
//         let parent = rng.gen_range(0..i);
//         let weight = rng.gen_range(1.0..10.0);
//         graph.add_edge(parent, i, weight);
//     }

//     // Add remaining random edges
//     let remaining_edges = m.saturating_sub(n - 1);
//     for _ in 0..remaining_edges {
//         let from = rng.gen_range(0..n);
//         let to = rng.gen_range(0..n);
//         if from != to {
//             let weight = rng.gen_range(1.0..10.0);
//             graph.add_edge(from, to, weight);
//         }
//     }

//     graph
// }

// fn create_dense_graph(n: usize, seed: u64) -> Graph {
//     let mut rng = StdRng::seed_from_u64(seed);
//     let mut graph = Graph::new(n);

//     // Create a dense graph with ~n^1.5 edges
//     let num_edges = ((n as f64).powf(1.5)) as usize;

//     for _ in 0..num_edges {
//         let from = rng.gen_range(0..n);
//         let to = rng.gen_range(0..n);
//         if from != to {
//             let weight = rng.gen_range(0.1..20.0);
//             graph.add_edge(from, to, weight);
//         }
//     }

//     graph
// }

// fn bench_dijkstra_vs_new_algorithm(c: &mut Criterion) {
//     let mut group = c.benchmark_group("SSSP Algorithms");

//     // Test on various graph sizes
//     let sizes = vec![50, 100, 200, 500, 1000];

//     for &n in &sizes {
//         // Sparse graphs (good for new algorithm)
//         let sparse_graph = create_sparse_graph(n, 2.0, 42);

//         group.bench_with_input(BenchmarkId::new("Dijkstra_Sparse", n), &n, |b, &_| {
//             b.iter(|| {
//                 let mut solver = SSSpSolver::new(sparse_graph.clone());
//                 black_box(solver.dijkstra(0, None));
//             })
//         });

//         group.bench_with_input(BenchmarkId::new("NewAlgorithm_Sparse", n), &n, |b, &_| {
//             b.iter(|| {
//                 black_box(dun_mao_all_distances(&sparse_graph, 0));
//             })
//         });
//     }

//     // Test on dense graphs (where Dijkstra might be better)
//     for &n in &[50, 100, 200] {
//         let dense_graph = create_dense_graph(n, 42);

//         group.bench_with_input(BenchmarkId::new("Dijkstra_Dense", n), &n, |b, &_| {
//             b.iter(|| {
//                 let mut solver = SSSpSolver::new(dense_graph.clone());
//                 black_box(solver.dijkstra(0, None));
//             })
//         });

//         group.bench_with_input(BenchmarkId::new("NewAlgorithm_Dense", n), &n, |b, &_| {
//             b.iter(|| {
//                 black_box(dun_mao_all_distances(&dense_graph, 0));
//             })
//         });
//     }

//     group.finish();
// }

// fn bench_scaling_behavior(c: &mut Criterion) {
//     let mut group = c.benchmark_group("Scaling Behavior");

//     // Test how algorithms scale with graph size on sparse graphs
//     let sizes = vec![100, 200, 400, 800, 1600];

//     for &n in &sizes {
//         let graph = create_sparse_graph(n, 1.5, 123);

//         group.bench_with_input(BenchmarkId::new("NewAlgorithm_Scaling", n), &n, |b, &_| {
//             b.iter(|| {
//                 black_box(dun_mao_all_distances(&graph, 0));
//             })
//         });
//     }

//     group.finish();
// }

// criterion_group!(
//     benches,
//     bench_dijkstra_vs_new_algorithm,
//     bench_scaling_behavior
// );
// criterion_main!(benches);
