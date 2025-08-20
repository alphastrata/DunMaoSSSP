#![cfg(all(
    feature = "flate2",
    feature = "bincode",
    feature = "log",
    feature = "env_logger"
))]

use fast_sssp::Graph;
use flate2::read::GzDecoder;
use log::info;
use reqwest::blocking::Client;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

const WIKI_TALK_URL: &str = "https://snap.stanford.edu/data/wiki-Talk.txt.gz";
const WIKI_TALK_FILENAME: &str = "wiki-Talk.txt.gz";
const OUTPUT_FILENAME: &str = "wiki-talk-graph.bin";

fn download_file(client: &Client, url: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading {}...", url);
    let mut response = client.get(url).send()?;
    let total_size = response.content_length().unwrap_or(0);

    println!("File size: {} bytes", total_size);
    let mut file = File::create(path)?;
    let mut downloaded = 0u64;
    let mut buffer = [0; 8192];

    while let Ok(n) = response.read(&mut buffer) {
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        downloaded += n as u64;

        if downloaded.is_multiple_of(1024 * 1024) {
            println!(
                "Downloaded: {:.1} MB",
                downloaded as f64 / (1024.0 * 1024.0)
            );
        }
    }

    println!(
        "Download complete: {:.1} MB",
        downloaded as f64 / (1024.0 * 1024.0)
    );
    Ok(())
}

fn parse_wiki_talk_to_graph(gz_path: &Path) -> Result<Graph, Box<dyn std::error::Error>> {
    info!("Parsing Wiki-Talk dataset from compressed file...");

    let file = File::open(gz_path)?;
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);

    let mut edges = Vec::new();
    let mut max_node = 0u32;
    let mut line_count = 0;

    // Skip header lines and parse edges
    for line_result in reader.lines() {
        let line = line_result?;

        // Skip comment lines (start with #)
        if line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2
            && let (Ok(u), Ok(v)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>())
        {
            edges.push((u, v));
            max_node = max_node.max(u).max(v);
            line_count += 1;

            if line_count % 100000 == 0 {
                println!("Parsed {} edges...", line_count);
            }
        }
    }

    println!("Found {} edges, max node ID: {}", edges.len(), max_node);

    // Create our Graph structure
    let mut graph = Graph::new((max_node + 1) as usize);

    println!("Building graph structure...");
    edges.iter().enumerate().for_each(|(i, (u, v))| {
        // Add edge with weight 1.0
        graph.add_edge(*u as usize, *v as usize, 1.0);

        if i % 100000 == 0 {
            println!("Added {} edges to graph...", i);
        }
    });

    info!(
        "Graph built with {} nodes and {} edges.",
        graph.vertices,
        graph.edge_count()
    );
    Ok(graph)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let data_dir = Path::new("data");
    if !data_dir.exists() {
        std::fs::create_dir_all(data_dir)?;
    }

    let wiki_talk_path = data_dir.join(WIKI_TALK_FILENAME);
    let output_path = data_dir.join(OUTPUT_FILENAME);

    // Check if we already have the processed graph
    if output_path.exists() {
        info!(
            "Wiki-Talk graph already processed at {}",
            output_path.display()
        );

        // Try to load and verify
        match Graph::from_file(&output_path) {
            Ok(graph) => {
                println!(
                    "Successfully loaded graph with {} vertices and {} edges",
                    graph.vertices,
                    graph.edge_count()
                );
                return Ok(());
            }
            Err(e) => {
                println!("Error loading existing graph: {}, will regenerate", e);
                std::fs::remove_file(&output_path)?;
            }
        }
    }

    // Download if needed
    if !wiki_talk_path.exists() {
        info!("Downloading Wiki-Talk dataset...");
        let client = Client::new();
        download_file(&client, WIKI_TALK_URL, wiki_talk_path.to_str().unwrap())?;
    } else {
        info!("Wiki-Talk dataset already downloaded.");
    }

    // Parse and convert
    info!("Processing Wiki-Talk dataset...");
    let graph = parse_wiki_talk_to_graph(&wiki_talk_path)?;

    // Save processed graph
    Graph::to_file(&graph, &output_path)?;

    println!("Successfully processed Wiki-Talk dataset!");
    println!(
        "Graph: {} vertices, {} edges",
        graph.vertices,
        graph.edge_count()
    );
    println!("Saved to: {}", output_path.display());

    Ok(())
}
