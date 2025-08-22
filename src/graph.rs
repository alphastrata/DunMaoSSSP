#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Graph {
    pub vertices: usize,
    pub edges: Vec<Vec<Edge>>,
}

impl Graph {
    pub fn new(vertices: usize) -> Self {
        Graph {
            vertices,
            edges: vec![Vec::new(); vertices],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges[from].push(Edge { to, weight });
    }

    pub fn edge_count(&self) -> usize {
        self.edges.iter().map(|adj| adj.len()).sum()
    }

    #[cfg(feature = "bincode")]
    pub fn from_file(path: &std::path::Path) -> Result<Graph, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let config = bincode::config::legacy();
        let reader = std::io::BufReader::new(file);
        let graph: Graph = bincode::decode_from_reader(reader, config)?;
        Ok(graph)
    }

    #[cfg(feature = "bincode")]
    pub fn to_file(
        graph: &Graph,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let config = bincode::config::legacy();
        let mut writer = std::io::BufWriter::new(file);

        bincode::serde::encode_into_std_write(graph, &mut writer, config)?;

        Ok(())
    }

    #[cfg(feature = "petgraph")]
    pub fn to_petgraph(&self) -> petgraph::Graph<usize, f64> {
        let mut pg_graph = petgraph::Graph::new();
        let nodes: Vec<_> = (0..self.vertices).map(|i| pg_graph.add_node(i)).collect();

        for (from_idx, from_node) in self.edges.iter().enumerate() {
            for edge in from_node {
                pg_graph.add_edge(nodes[from_idx], nodes[edge.to], edge.weight);
            }
        }
        pg_graph
    }
}
