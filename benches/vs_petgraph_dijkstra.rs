// #![allow(dead_code)] // The file is not used by all tests, so this is needed.

// #[path = "../tests/graph_loader.rs"]
// mod graph_loader;

// use criterion::{criterion_group, criterion_main, Criterion};
// use fast_sssp::dun_mao_all_distances;
// use graph_loader::{read_dimacs_graph_for_fast_sssp, read_dimacs_graph_for_petgraph};
// use petgraph::algo::dijkstra;
// use std::path::Path;

// fn benchmark_solvers(c: &mut Criterion) {
//     let mut group = c.benchmark_group("SSSP Solvers on Rome99");

//     // Benchmark fast_sssp
//     group.bench_function("fast_sssp", |b| {
//         let graph = read_dimacs_graph_for_fast_sssp(Path::new("tests/test_data/Rome99"));
//         b.iter(|| {
//             dun_mao_all_distances(&graph, 0);
//         })
//     });

//     // Benchmark petgraph's Dijkstra
//     group.bench_function("petgraph_dijkstra", |b| {
//         let (graph, node_map) = read_dimacs_graph_for_petgraph(Path::new("tests/test_data/Rome99"));
//         let source_node = node_map[&1];
//         b.iter(|| {
//             dijkstra(&graph, source_node, None, |e| *e.weight());
//         })
//     });

//     group.finish();
// }

// criterion_group!(benches, benchmark_solvers);
// criterion_main!(benches);
