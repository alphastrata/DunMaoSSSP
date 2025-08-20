# A Fast SSSP

> BMSSP (Bounded Multi-Source Shortest Path) or? Dun-Mal et al? I dunno what they want to call it.

<!-- [![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/) -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A naive Rust implementation of the _breakthrough?_ deterministic algorithm for Single-Source Shortest Paths (SSSP) that breaks the O(m + n log n) sorting barrier on directed graphs. This is based on the paper ["Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"](https://arxiv.org/abs/2504.17033) by Duan, Mao, Mao, Shu, and Yin (2025).

It achieves **O(m log^(2/3) n)** time complexity for SSSP on directed graphs with real non-negative edge weights in the comparison-addition model.

NOTE:
This is more of a POC than a functional library for use in your own code, the 'data' you'd likely want available in a Graph's `Node` type etc is not set up here, I implemented this because, implementing papers is fun and what I do on my weekends.

### Paper claims:

- **Time Complexity**: O(m log^(2/3) n)
- **Space Complexity**: O(n + m)
- **Best for**: Sparse directed graphs where breaking the sorting barrier matters
- **Practical use**: Currently more of theoretical interest; Dijkstra may be faster in practice

### Tests

```bash
# Run all tests
cargo test -F full
```

Extended testing uses data from [here](https://www.diag.uniroma1.it/~challenge9/) and from `snap.stanford`

> # So... Is it good?.. it's complicated.

### Benchmarking:

#### Get data:

```bash
cargo run --release --bin fetch_data -F full # Will download the wikipedia-talk dataset for you
```

//TODO:
or:
## LiveJournal (best bet to SEE stuff improve)
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz

## Pokec (backup)  
wget https://snap.stanford.edu/data/soc-Pokec-relationships.txt.gz

## YouTube (smaller test)
wget https://snap.stanford.edu/data/com-youtube.ungraph.txt.gz
if you want to run the benchmarks, slap them in `./data`

```bash
cargo run bench -F full # synthetics
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- **Primary Paper**: ["Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"](https://arxiv.org/abs/2504.17033) by Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin (2025)

<!-- ```
@article{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  journal={arXiv preprint arXiv:2504.17033},
  year={2025}
}
``` -->

## TODOs:

- \[\] a `Node` and or a `Weight` would need to be able to carry a wider variety of data types to be useful..
- [] make the API like Petgraph (which is rather nice and well thought out...)