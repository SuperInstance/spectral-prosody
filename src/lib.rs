//! # Spectral Prosody
//!
//! Applies spectral graph theory to metrical patterns across languages.
//! Poetry traditions have spectral fingerprints via graph Laplacian eigenvalues.
//! Independent cultures converge on isospectral meters.

pub mod metrical_graph;
pub mod laplacian_scan;
pub mod rhyme_adjacency;
pub mod tradition_embedding;
pub mod iso_breath;
pub mod dial_scan;
pub mod linalg;

pub use metrical_graph::{MetricalLine, MetricalGraph};
pub use laplacian_scan::SpectralSignature;
pub use rhyme_adjacency::RhymeScheme;
pub use tradition_embedding::TraditionEmbedding;
