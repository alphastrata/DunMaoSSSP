use criterion::{Criterion, criterion_group, criterion_main};
use fast_sssp::{DuanMaoSolverV2, Graph};
use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::hint::black_box;
use std::path::Path;

#[path = "../tests/graph_loader.rs"]
mod graph_loader;

fn run_duan_mao_v2(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = DuanMaoSolverV2::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn run_petgraph_dijkstra(
    graph: &DiGraph<(), f64>,
    node_map: &HashMap<usize, NodeIndex>,
    pairs: &[(usize, usize)],
) {
    for (source, goal) in pairs {
        if let (Some(&source_node), Some(&goal_node)) = (node_map.get(source), node_map.get(goal)) {
            black_box(dijkstra(&graph, source_node, Some(goal_node), |e| {
                *e.weight()
            }));
        }
    }
}

fn benchmark(c: &mut Criterion) {
    let live_journal_path = Path::new("data/soc-LiveJournal1.txt.gz");
    if !live_journal_path.exists() {
        println!("LiveJournal dataset not found, skipping benchmark.");
        return;
    }
    let (petgraph_graph, node_map) = graph_loader::read_wiki_graph_for_petgraph(live_journal_path);
    let fast_sssp_graph = graph_loader::read_wiki_graph_for_fast_sssp(live_journal_path);

    let mut rng = rand::rng();
    let mut nodes: Vec<usize> = (0..fast_sssp_graph.vertices).collect();
    nodes.shuffle(&mut rng);
    let pairs: Vec<(usize, usize)> = nodes
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .take(5) // keep the number of pairs reasonable!
        .collect();

    let mut group = c.benchmark_group("livejournal SSSP (V2)");
    group.sample_size(10); // In particular on this bench, the DuanMao can really be a lot faster than the Petgraph comparison -- so to keep you from having to set at the computer for more than 10 minutes...

    group.bench_function("duan_mao_v2", |b| {
        b.iter(|| run_duan_mao_v2(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.bench_function("petgraph_dijkstra", |b| {
        b.iter(|| {
            run_petgraph_dijkstra(
                black_box(&petgraph_graph),
                black_box(&node_map),
                black_box(&pairs),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
