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
cargo test

# Run with output
cargo test -- --nocapture

```

Extended testing uses data from [here](https://www.diag.uniroma1.it/~challenge9/).

> # So... Is it good? -- Yeah, it shows promise!

### Benchmarking:

#### Get data:

```bash
cargo run --release --bin fetch_data -F full # Will download the wikipedia-talk dataset for you
```

```bash
cargo run bench # synthetics
# or
cargo run --release --example wiki_talk_benchmark -F bincode # It'll error out if you don't have the data
```

Figures are from my ancient 2013 Macbook Pro with an Intel i7-4960HQ and 16GB od DDR3 (garbage machine I know...)

#### Real-World:

\_ Wiki-Talk \_

```sh
Benchmarking on Wiki-Talk dataset:
Source   Dijkstra (ms)   New Algo (ms)   Sp          
-------------------------------------------------------
0        102             102             1.00        x
100      107680          109             987.89      x
1000     78398           115             681.72      x
5000     68606           100             686.06      x
```

#### Synthetics:

| Benchmark | Best, Avg, Worst | Outliers |
| :--- | :--- | :--- |
| **SSSP Algorithms/Dijkstra_Sparse/50** | `[9.0217 µs 9.1254 µs 9.2647 µs]` | 6 outliers (6.00%): 3 high mild, 3 high severe |
| **SSSP Algorithms/NewAlgorithm_Sparse/50** | `[7.8958 µs 7.9676 µs 8.0494 µs]` | 4 outliers (4.00%): 3 high mild, 1 high severe |
| **SSSP Algorithms/Dijkstra_Sparse/100** | `[41.739 µs 42.105 µs 42.573 µs]` | 10 outliers (10.00%): 5 high mild, 5 high severe |
| **SSSP Algorithms/NewAlgorithm_Sparse/100**| `[14.838 µs 15.773 µs 17.250 µs]` | 8 outliers (8.00%): 5 high mild, 3 high severe |
| **SSSP Algorithms/Dijkstra_Sparse/200** | `[144.47 µs 149.85 µs 158.39 µs]` | 5 outliers (5.00%): 2 high mild, 3 high severe |
| **SSSP Algorithms/NewAlgorithm_Sparse/200**| `[28.600 µs 29.919 µs 31.739 µs]` | 5 outliers (5.00%): 4 high mild, 1 high severe |
| **SSSP Algorithms/Dijkstra_Sparse/500** | `[952.55 µs 976.34 µs 1.0029 ms]` | 10 outliers (10.00%): 8 high mild, 2 high severe |
| **SSSP Algorithms/NewAlgorithm_Sparse/500**| `[70.068 µs 73.532 µs 77.810 µs]` | 7 outliers (7.00%): 4 high mild, 3 high severe |
| **SSSP Algorithms/Dijkstra_Sparse/1000** | `[4.5097 ms 4.6387 ms 4.8132 ms]` | 10 outliers (10.00%): 9 high mild, 1 high severe |
| **SSSP Algorithms/NewAlgorithm_Sparse/1000**| `[134.19 µs 142.28 µs 154.94 µs]` | 8 outliers (8.00%): 5 high mild, 3 high severe |
| **SSSP Algorithms/Dijkstra_Dense/50** | `[29.972 µs 30.786 µs 31.723 µs]` | 7 outliers (7.00%): 6 high mild, 1 high severe |
| **SSSP Algorithms/NewAlgorithm_Dense/50** | `[10.006 µs 10.386 µs 10.884 µs]` | 8 outliers (8.00%): 4 high mild, 4 high severe |
| **SSSP Algorithms/Dijkstra_Dense/100** | `[236.45 µs 244.29 µs 254.07 µs]` | 5 outliers (5.00%): 2 high mild, 3 high severe |
| **SSSP Algorithms/NewAlgorithm_Dense/100**| `[17.637 µs 17.806 µs 18.023 µs]` | 9 outliers (9.00%): 5 high mild, 4 high severe |
| **SSSP Algorithms/Dijkstra_Dense/200** | `[988.78 µs 994.59 µs 1.0006 ms]` | 2 outliers (2.00%): 2 high mild |
| **SSSP Algorithms/NewAlgorithm_Dense/200**| `[35.157 µs 35.351 µs 35.571 µs]` | 5 outliers (5.00%): 3 high mild, 2 high severe |
| **Scaling Behavior/NewAlgorithm_Scaling/100**| `[12.204 µs 12.297 µs 12.398 µs]` | 7 outliers (7.00%): 5 high mild, 2 high severe |
| **Scaling Behavior/NewAlgorithm_Scaling/200**| `[26.112 µs 26.596 µs 27.194 µs]` | 8 outliers (8.00%): 3 high mild, 5 high severe |
| **Scaling Behavior/NewAlgorithm_Scaling/400**| `[48.366 µs 50.270 µs 53.218 µs]` | 9 outliers (9.00%): 6 high mild, 3 high severe |
| **Scaling Behavior/NewAlgorithm_Scaling/800**| `[92.562 µs 96.457 µs 101.63 µs]` | 9 outliers (9.00%): 5 high mild, 4 high severe |
| **Scaling Behavior/NewAlgorithm_Scaling/1600**|`[190.82 µs 195.76 µs 202.02 µs]`| 6 outliers (6.00%): 2 high mild, 4 high severe |

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

- \[\] workspace & feature flag deps like `serde` etc to clean up deps
- \[\] bring in some other libraries that have a Dijkstra in them, and bench against this.
- \[\] a `Node` and or a `Weight` would need to be able to carry a wider variety of data types to be useful..