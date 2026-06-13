# spectral-prosody

**Spectral graph theory applied to metrical patterns across languages.**

Poetry traditions have spectral fingerprints — the eigenvalues of graph Laplacians constructed from metrical structure reveal deep structural properties. Independent cultures converge on isospectral meters, suggesting universal constraints from breath and cognition.

## Modules

| Module | Description |
|---|---|
| `metrical_graph` | Construct weighted graphs from poetic corpora. Nodes = lines, edges = metrical similarity, weights = syllable distance. |
| `laplacian_scan` | Compute graph Laplacian eigenvalues for poetic traditions. Each tradition = point in spectral space. |
| `rhyme_adjacency` | Encode rhyme schemes as adjacency matrices. ABAB, AABB, ABBA, and free verse produce distinct spectral radii. |
| `tradition_embedding` | Low-dimensional spectral embeddings of poetic traditions. Visualize the "space of all poetry". |
| `iso_breath` | Test the conjecture that breath-constrained meters produce isospectral graphs regardless of language. |
| `dial_scan` | Connect to dial-theory: spectral distance between traditions as a dial dimension. |
| `linalg` | Gaussian elimination and Jacobi eigenvalue method from scratch. No external math dependencies. |

## Core Types

```rust
use spectral_prosody::*;

// A line of poetry with metrical information
let line = MetricalLine::new(
    vec![1.0; 10],                                        // syllable durations
    vec![false, true, false, true, false, true, false, true, false, true], // iambic stress
    "English".into(),
);

// Build a graph from multiple lines
let graph = MetricalGraph::from_lines(&lines);

// Extract spectral signature
let signature = SpectralSignature::from_graph(&graph, "English Iambic Pentameter");
println!("Fiedler value: {}", signature.fiedler_value());
println!("Spectral radius: {}", signature.spectral_radius());

// Classify rhyme schemes
let abab = RhymeScheme::from_str("ABAB");
println!("Spectral radius: {}", abab.spectral_radius());

// Compare traditions spectrally
let distance = sig_en.distance_to(&sig_fr);
let similarity = sig_en.cosine_similarity(&sig_fr);
```

## Mathematical Foundation

### Graph Construction
Lines of poetry become nodes. Edge weights encode metrical similarity:

```
w(i,j) = 0.6 × stress_similarity(i,j) + 0.4 × syllable_distance(i,j)
```

### Laplacian Eigenvalues
The unnormalized graph Laplacian `L = D - A` encodes metrical structure in its spectrum:
- **λ₁ = 0** (always, for connected components)
- **λ₂ (Fiedler value)** = algebraic connectivity, measures metrical coherence
- **λ_max** = spectral radius, relates to rhythmic complexity

### Jacobi Eigenvalue Method
Eigenvalues are computed via the Jacobi iterative method — rotations annihilate off-diagonal entries of the symmetric Laplacian matrix. Converges for all real symmetric matrices.

### Cheeger Constant
The Cheeger constant `h(G)` measures the "bottleneck" of the metrical graph:

```
h²/2 ≤ λ₂ ≤ 2h
```

Approximated as `h ≈ √(2λ₂)`.

### Iso-Breath Conjecture
Meters constrained by human breath capacity (~1 breath unit ≈ 10-16 syllables) produce isospectral graph structures regardless of language family. Iambic pentameter (English, 10 syllables), Sanskrit sloka (16 syllables), and French alexandrine (12 syllables) all inhabit a narrow region of spectral space.

### Dial Theory
Each eigenvalue index becomes a "dial dimension" along which traditions vary. The full dial-space distance between traditions captures structural divergence beyond simple syllable counting.

## Testing

```bash
cargo test
```

65 tests covering:
- Metrical graph construction and Laplacian properties
- Jacobi eigenvalue computation (identity, 2×2, 3×3 path graph)
- Gaussian elimination (identity, general, singular)
- Rhyme scheme spectral classification (ABAB, AABB, ABBA, free verse)
- Spectral clustering and tradition classification
- Cheeger constant computation
- Random walk traversal time (Kemeny's constant)
- Tradition embedding and dimensionality reduction
- Iso-breath conjecture validation
- Dial-space construction and proximity ranking
- Serde roundtrips for all public types

## Dependencies

- `serde` — serialization for all public types
- `serde_json` — test-only JSON roundtrips

No external math libraries. All linear algebra implemented from scratch.

## License

MIT
