# A Fast SSSP

> BMSSP (Bounded Multi-Source Shortest Path) or? Duan-Mao et al? I dunno what they want to call it.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust implementation of the breakthrough deterministic algorithm for Single-Source Shortest Paths (SSSP) that breaks the O(m + n log n) sorting barrier on directed graphs. This is based on the paper ["Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"](https://arxiv.org/abs/2504.17033) by Duan, Mao, Mao, Shu, and Yin (2025).

It achieves **O(m log^(2/3) n)** time complexity for SSSP on directed graphs with real non-negative edge weights in the comparison-addition model.

**NOTE:** This is more of a POC than a functional library. The data structures you'd likely want in a Graph's `Node` type etc are not set up here. I implemented this because implementing papers is fun and what I do on my weekends.

## Algorithm Properties

- **Time Complexity**: O(m log^(2/3) n)
- **Space Complexity**: O(n + m)
- **First deterministic algorithm** to break the O(m + n log n) barrier even for undirected graphs
- **Best for**: Large sparse directed graphs where breaking the sorting barrier matters
- **Model**: Works in comparison-addition model (only comparisons and additions on edge weights)

## Performance Results

The algorithm can be **very, very good** on appropriate graphs:

```
livejournal SSSP (V2)/duan_mao_v2
                        time:   [583.02 ms 588.80 ms 594.90 ms]
livejournal SSSP (V2)/petgraph_dijkstra
                        time:   [8.8686 s 9.0271 s 9.1969 s]
```
> Test rig: Kernel: 6.12.10-76061203-generic with CPU: AMD Ryzen 9 5950X (32) @ 5.084GHz 

**That's ~15x faster than Dijkstra on LiveJournal dataset!**

## When to Use This Algorithm (If you really must.. but you probably shouldn't)

### ✅ Use When:
- **Large sparse directed graphs**: n > 10⁵, m ≈ n to n log n
- **Repeated computations**: You solve SSSP frequently on similar graphs  
- **Graph density ratio m/n is low**: Algorithm shines when m ≈ n

### ❌ Avoid When:
- **Small graphs**: n < 10³ (Dijkstra ~will likely~ be faster)
- **Dense graphs**: m ≈ n² (benefit diminishes significantly)
- **One-off computations** on moderately sized graphs
- **Implementation time is limited**: Dijkstra is much simpler
- **Need shortest path tree structure**: This algorithm focuses on distances only

### Suggested Decision Framework

```
1. Is n < 1,000?
   → Use Dijkstra's (simpler, likely faster in practice), there's many many many SSSP algos do the reading they all have tradeoffs!

2. Is m > n^1.5?
   → Consider whether theoretical benefit justifies complexity

3. Is this a one-off computation?
   → Stick with Dijkstra's unless n > 10^6, even then you can run a big-ass graph in the time it takes to implement and test something is giving you valid output!

4. Do you solve many similar SSSP problems?
   → Implementation cost amortizes; consider for n > 10^4

5. Research/experimental context?
   → Go work out more of these!
```

**Practical threshold**: Most beneficial when n > 10⁴ and graph is relatively sparse (m ≤ n log n).

## Requirements

Your graph must be:
- ✅ **Directed** (required)
- ✅ **Non-negative real edge weights** (required)
- ✅ **Single-source** (not all-pairs)

## Usage

### Tests
```bash
# Run all tests
cargo test -F full
```

### Get benchmark data:
```bash
# Download wikipedia-talk dataset
cargo run --release --bin fetch_data -F full

# Or manually get other datasets:
# LiveJournal (best for seeing improvements)
wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz

# Pokec (backup)  
wget https://snap.stanford.edu/data/soc-Pokec-relationships.txt.gz

# YouTube (smaller test)
wget https://snap.stanford.edu/data/com-youtube.ungraph.txt.gz
```

Place datasets in `./data` directory.

### Run benchmarks:
```bash
cargo run bench -F full
```

## Implementation Notes

This implementation includes:
- **FindPivots algorithm**: Reduces frontier size by factor of log^Ω(1)(n)
- **Block-based data structure**: From Lemma 3.3 supporting efficient Insert/BatchPrepend/Pull operations
- **Recursive partitioning**: O(log n/t) levels with t = ⌊log^(2/3)(n)⌋
- **Constant-degree transformation**: Converts arbitrary graphs as described in the paper

**Complexity vs. Benefit**: The big-O notation hides potentially large constant factors. The paper mentions "large constant C" in analysis, so practical benefits depend heavily on your specific graph characteristics.

## TODOs

- [ ] Generic `Node` and `Weight` types to carry wider variety of data
- [ ] Make API more like Petgraph (which is rather nice and well thought out...)
- [ ] Better error handling and edge cases
- [ ] More comprehensive benchmarking suite
- [ ] Reach a conclusion about v1 vs v2 
- [ ] See if the parallel numbers can shine even brighter against the sequential v2 version.

## License

MIT

## References

- **Primary Paper**: ["Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"](https://arxiv.org/abs/2504.17033) by Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin (2025)

