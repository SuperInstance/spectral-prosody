# spectral-prosody

[![crates.io](https://img.shields.io/crates/v/spectral-prosody.svg)](https://crates.io/crates/spectral-prosody)
[![docs.rs](https://docs.rs/spectral-prosody/badge.svg)](https://docs.rs/spectral-prosody)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

Speech and music have rhythm at multiple scales simultaneously: macro rhythm (phrase structure), meso rhythm (beat patterns), and micro rhythm (individual note/syllable timing). These layers interact — a sentence's phrase structure constrains where accents fall, which constrains syllable timing.

Most prosody analyzers use time-domain methods (autocorrelation, onset detection). These find one dominant period and miss the layered structure. Spectral graph methods capture all layers at once.

## The Idea

Build a graph where nodes are acoustic events (beats, syllables, notes) and edges connect temporally proximate events with weights based on feature similarity (pitch, energy, spectral centroid). The **graph Laplacian's eigenvalues** then decompose the rhythm into layers:

- **Low eigenvalues** = slow-varying component (macro rhythm, phrase structure)
- **Mid eigenvalues** = the primary beat pattern
- **High eigenvalues** = fast-varying micro-timing (swing, groove)

The **Fiedler vector** (eigenvector for λ₂) naturally segments the prosody into phrases — sign changes in the Fiedler vector correspond to phrase boundaries.

## How It Works

### Build the prosody graph

```rust
use spectral_prosody::{ProsodyNode, ProsodyGraph};

let nodes = vec![
    ProsodyNode { time: 0.0, energy: 0.8, pitch: 220.0, duration: 0.15, spectral_centroid: 1500.0 },
    ProsodyNode { time: 0.3, energy: 0.6, pitch: 330.0, duration: 0.12, spectral_centroid: 1800.0 },
    // ... more nodes
];

// k-NN graph with Gaussian kernel weighting
let graph = ProsodyGraph::from_nodes(&nodes, /* k */ 5);
```

### Extract rhythmic layers

```rust
use spectral_prosody::RhythmExtractor;

let extractor = RhythmExtractor::new(&graph);
let layers = extractor.extract(/* max_layers */ 5);

for (i, layer) in layers.iter().enumerate() {
    println!("Layer {}: period={:.2}s strength={:.3}",
        i, layer.period, layer.strength);
}
// Layer 0: period=4.0s strength=0.92  (phrase level)
// Layer 1: period=0.5s strength=0.87  (beat level)
// Layer 2: period=0.25s strength=0.31 (sub-beat)
```

### Segment into phrases

```rust
use spectral_prosody::PhraseSegmenter;

let segmenter = PhraseSegmenter::new(&graph);
let phrases = segmenter.segment(&layers[0]);

for phrase in &phrases {
    println!("Phrase [{}..{}]: avg_energy={:.2}",
        phrase.start_idx, phrase.end_idx, phrase.average_energy);
}
```

### Extract features from raw data

```rust
use spectral_prosody::ProsodyFeature;

let pitch = ProsodyFeature::from_pitch(&audio_samples, /* sample_rate */ 44100);
let energy = ProsodyFeature::from_energy(&audio_samples, /* window_ms */ 25);
let centroid = ProsodyFeature::from_spectral_centroid(&audio_samples, 44100);
```

### Export to MIDI

```rust
use spectral_prosody::midi::MidiExporter;

let midi = MidiExporter::from_layers(&layers, /* bpm */ 120);
let bytes = midi.to_bytes(); // Standard MIDI file
```

## Module Map

| Module | What it does |
|---|---|
| `prosody` | `ProsodyGraph` — build k-NN or fully-connected graph from prosody nodes |
| `rhythm` | `RhythmExtractor` — eigen-decompose Laplacian → rhythmic layers |
| `phrase` | `PhraseSegmenter` — Fiedler vector → phrase boundaries |
| `feature` | `ProsodyFeature` — extract timing, energy, pitch, spectral centroid |
| `midi` | `MidiExporter` — rhythmic layers → MIDI patterns |
| `error` | `ProsodyError` |

## Why Graphs, Not FFT?

FFT decomposes a signal into sinusoidal frequencies. But rhythm isn't sinusoidal — it's event-based. A graph naturally represents the *relationship* between events, and the graph Laplacian's spectrum captures periodicity in the **structure**, not the waveform. A rhythm that's perfectly regular but has varying amplitudes is one clean eigenvalue in graph space but a messy harmonic series in FFT space.

## Links

- [Documentation](https://docs.rs/spectral-prosody)
- [Repository](https://github.com/SuperInstance/spectral-prosody)
- [crates.io](https://crates.io/crates/spectral-prosody)

## License

MIT
