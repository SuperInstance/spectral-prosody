//! # Spectral Prosody
//!
//! Applies spectral graph theory to metrical patterns across languages.
//! Poetry traditions have spectral fingerprints via graph Laplacian eigenvalues.
//! Independent cultures converge on isospectral meters.

// Pre-existing numeric code triggers several clippy pedantic lints.
#![allow(
    unused_imports,
    clippy::needless_range_loop,
    clippy::should_implement_trait,
    clippy::manual_clamp
)]

pub mod dial_scan;
pub mod iso_breath;
pub mod laplacian_scan;
pub mod linalg;
pub mod metrical_graph;
pub mod rhyme_adjacency;
pub mod tradition_embedding;

pub use laplacian_scan::SpectralSignature;
pub use metrical_graph::{MetricalGraph, MetricalLine};
pub use rhyme_adjacency::RhymeScheme;
pub use tradition_embedding::TraditionEmbedding;
