//! # spectral-prosody
//!
//! Spectral graph methods for rhythmic prosody analysis.
//!
//! This crate maps speech/music prosody to a graph where nodes represent beats or syllables
//! and edges represent temporal proximity. The graph Laplacian's spectrum reveals rhythmic
//! patterns: low eigenvalues correspond to macro rhythm (large-scale phrasing), while high
//! eigenvalues correspond to micro rhythm (fine-grained beat structure). Spectral clustering
//! segments prosody into phrases.
//!
//! ## Quick Start
//!
//! ```rust
//! use spectral_prosody::{ProsodyNode, ProsodyGraph, RhythmExtractor, PhraseSegmenter};
//!
//! // Create prosody nodes from timing/energy/pitch data
//! let nodes: Vec<ProsodyNode> = (0..8)
//!     .map(|i| ProsodyNode::new(i as f64 * 0.5, 1.0, 220.0 + i as f64 * 5.0, 0.25, 3000.0))
//!     .collect();
//!
//! // Build a k-nearest-neighbor graph
//! let graph = ProsodyGraph::build_knn(nodes, 3, 1.0).unwrap();
//!
//! // Extract rhythmic layers via spectral decomposition
//! let extractor = RhythmExtractor::new(5);
//! let layers = extractor.extract(&graph).unwrap();
//!
//! // Segment into phrases using the Fiedler vector
//! let segmenter = PhraseSegmenter::new(4);
//! let phrases = segmenter.segment(&graph).unwrap();
//! ```

pub mod error;
pub mod feature;
pub mod midi;
pub mod phrase;
pub mod prosody;
pub mod rhythm;

pub use error::ProsodyError;
pub use feature::{ProsodyFeature, estimate_pitch};
pub use midi::{MidiNote, frequency_to_midi, midi_to_frequency, layers_to_midi, notes_to_csv};
pub use phrase::{Phrase, PhraseSegmenter};
pub use prosody::{ProsodyGraph, ProsodyNode, graph_from_features};
pub use rhythm::{RhythmExtractor, RhythmLayer, algebraic_connectivity};
