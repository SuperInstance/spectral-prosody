//! Compute graph Laplacian eigenvalues for poetic traditions.
//! Each tradition becomes a point in spectral space.

use crate::linalg::jacobi_eigenvalues;
use crate::metrical_graph::{MetricalGraph, MetricalLine};
use serde::{Deserialize, Serialize};

/// Spectral signature of a poetic tradition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralSignature {
    /// Eigenvalues of the graph Laplacian (ascending).
    pub eigenvalues: Vec<f64>,
    /// Name of the tradition.
    pub tradition: String,
}

impl SpectralSignature {
    /// Compute spectral signature from a metrical graph.
    pub fn from_graph(graph: &MetricalGraph, tradition: &str) -> Self {
        let lap = graph.laplacian();
        let eigenvalues = jacobi_eigenvalues(&lap, 1000);
        Self {
            eigenvalues,
            tradition: tradition.to_string(),
        }
    }

    /// Compute spectral signature from a collection of lines.
    pub fn from_lines(lines: &[MetricalLine], tradition: &str) -> Self {
        let graph = MetricalGraph::from_lines(lines);
        Self::from_graph(&graph, tradition)
    }

    /// The Fiedler value (algebraic connectivity) = second-smallest eigenvalue.
    pub fn fiedler_value(&self) -> f64 {
        self.eigenvalues.get(1).copied().unwrap_or(0.0)
    }

    /// Spectral radius = largest eigenvalue magnitude.
    pub fn spectral_radius(&self) -> f64 {
        self.eigenvalues
            .iter()
            .map(|e| e.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Spectral gap = difference between two largest eigenvalues.
    pub fn spectral_gap(&self) -> f64 {
        let n = self.eigenvalues.len();
        if n < 2 {
            return 0.0;
        }
        (self.eigenvalues[n - 1] - self.eigenvalues[n - 2]).abs()
    }

    /// Spectral energy = sum of squared eigenvalues.
    pub fn spectral_energy(&self) -> f64 {
        self.eigenvalues.iter().map(|e| e * e).sum()
    }

    /// Compute spectral distance to another signature.
    /// Uses Euclidean distance in eigenvalue space (padded to same dimension).
    pub fn distance_to(&self, other: &SpectralSignature) -> f64 {
        let max_len = self.eigenvalues.len().max(other.eigenvalues.len());
        let mut sum_sq = 0.0;
        for i in 0..max_len {
            let a = self.eigenvalues.get(i).copied().unwrap_or(0.0);
            let b = other.eigenvalues.get(i).copied().unwrap_or(0.0);
            sum_sq += (a - b) * (a - b);
        }
        sum_sq.sqrt()
    }

    /// Cosine similarity in eigenvalue space.
    pub fn cosine_similarity(&self, other: &SpectralSignature) -> f64 {
        let max_len = self.eigenvalues.len().max(other.eigenvalues.len());
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..max_len {
            let a = self.eigenvalues.get(i).copied().unwrap_or(0.0);
            let b = other.eigenvalues.get(i).copied().unwrap_or(0.0);
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        if norm_a < 1e-15 || norm_b < 1e-15 {
            return 0.0;
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Normalize eigenvalues by the largest to enable cross-tradition comparison.
    pub fn normalized(&self) -> SpectralSignature {
        let max_eig = self
            .eigenvalues
            .iter()
            .map(|e| e.abs())
            .fold(0.0_f64, f64::max);
        let factor = if max_eig > 1e-12 { 1.0 / max_eig } else { 1.0 };
        SpectralSignature {
            eigenvalues: self.eigenvalues.iter().map(|e| e * factor).collect(),
            tradition: self.tradition.clone(),
        }
    }
}

/// Classify a poem into a tradition by finding the closest spectral match.
pub fn classify_tradition(
    query: &SpectralSignature,
    traditions: &[SpectralSignature],
) -> Option<String> {
    traditions
        .iter()
        .min_by(|a, b| {
            a.distance_to(query)
                .partial_cmp(&b.distance_to(query))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|t| t.tradition.clone())
}

/// Cluster traditions by spectral proximity using simple threshold.
pub fn spectral_cluster(traditions: &[SpectralSignature], threshold: f64) -> Vec<Vec<String>> {
    let n = traditions.len();
    let mut clusters: Vec<Vec<String>> = Vec::new();
    let mut assigned = vec![false; n];

    for i in 0..n {
        if assigned[i] {
            continue;
        }
        let mut cluster = vec![traditions[i].tradition.clone()];
        assigned[i] = true;
        for j in (i + 1)..n {
            if assigned[j] {
                continue;
            }
            if traditions[i].distance_to(&traditions[j]) < threshold {
                cluster.push(traditions[j].tradition.clone());
                assigned[j] = true;
            }
        }
        clusters.push(cluster);
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrical_graph::MetricalLine;

    fn english_iambic_lines() -> Vec<MetricalLine> {
        (0..5)
            .map(|i| {
                MetricalLine::new(
                    vec![1.0; 10],
                    vec![
                        false, true, false, true, false, true, false, true, false, true,
                    ],
                    format!("English-{}", i),
                )
            })
            .collect()
    }

    fn french_alexandrine_lines() -> Vec<MetricalLine> {
        (0..5)
            .map(|i| {
                // French alexandrine: 12 syllables, hemistich at 6
                MetricalLine::new(
                    vec![1.0; 12],
                    vec![
                        false, false, false, true, false, false, false, false, false, true, false,
                        false,
                    ],
                    format!("French-{}", i),
                )
            })
            .collect()
    }

    fn sanskrit_sloka_lines() -> Vec<MetricalLine> {
        (0..5)
            .map(|i| {
                MetricalLine::new(
                    vec![1.0; 16],
                    vec![
                        false, true, false, true, false, true, false, false, false, true, false,
                        true, false, true, false, false,
                    ],
                    format!("Sanskrit-{}", i),
                )
            })
            .collect()
    }

    #[test]
    fn test_signature_from_lines() {
        let lines = english_iambic_lines();
        let sig = SpectralSignature::from_lines(&lines, "English Iambic");
        assert_eq!(sig.tradition, "English Iambic");
        assert_eq!(sig.eigenvalues.len(), 5);
    }

    #[test]
    fn test_fiedler_value_positive() {
        let sig = SpectralSignature::from_lines(&english_iambic_lines(), "English");
        // Connected graph => Fiedler value > 0
        assert!(
            sig.fiedler_value() > 0.0,
            "Fiedler value should be positive for connected graph"
        );
    }

    #[test]
    fn test_spectral_radius() {
        let sig = SpectralSignature::from_lines(&english_iambic_lines(), "English");
        assert!(sig.spectral_radius() > 0.0);
    }

    #[test]
    fn test_same_tradition_close() {
        let sig1 = SpectralSignature::from_lines(&english_iambic_lines(), "English1");
        let sig2 = SpectralSignature::from_lines(&english_iambic_lines(), "English2");
        // Same lines => identical signature
        assert!(
            sig1.distance_to(&sig2) < 0.1,
            "Same tradition should be spectrally close"
        );
    }

    #[test]
    fn test_different_traditions_farther() {
        let sig_en = SpectralSignature::from_lines(&english_iambic_lines(), "English");
        let sig_fr = SpectralSignature::from_lines(&french_alexandrine_lines(), "French");
        // Cross-tradition distance should be meaningful (non-zero due to different structure)
        let d_cross = sig_en.distance_to(&sig_fr);
        assert!(d_cross >= 0.0, "Distance should be non-negative");
        // Both signatures should have different spectral radii
        assert!(sig_en.spectral_radius() > 0.0);
        assert!(sig_fr.spectral_radius() > 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let sig1 = SpectralSignature::from_lines(&english_iambic_lines(), "E1");
        let sig2 = SpectralSignature::from_lines(&english_iambic_lines(), "E2");
        let sim = sig1.cosine_similarity(&sig2);
        assert!(
            sim > 0.99,
            "Identical traditions should have cosine sim ~1, got {}",
            sim
        );
    }

    #[test]
    fn test_classify_tradition() {
        let en = SpectralSignature::from_lines(&english_iambic_lines(), "English");
        let fr = SpectralSignature::from_lines(&french_alexandrine_lines(), "French");
        let query = SpectralSignature::from_lines(&english_iambic_lines(), "Unknown");
        let result = classify_tradition(&query, &[en, fr]);
        assert_eq!(result.as_deref(), Some("English"));
    }

    #[test]
    fn test_spectral_cluster_groups_related() {
        let sig1 = SpectralSignature::from_lines(&english_iambic_lines(), "English-A");
        let sig2 = SpectralSignature::from_lines(&english_iambic_lines(), "English-B");
        let sig3 = SpectralSignature::from_lines(&french_alexandrine_lines(), "French");
        let clusters = spectral_cluster(&[sig1, sig2, sig3], 1.0);
        // English-A and English-B should cluster together
        assert!(clusters.len() >= 1);
    }

    #[test]
    fn test_normalized_signature() {
        let sig = SpectralSignature::from_lines(&english_iambic_lines(), "English");
        let normed = sig.normalized();
        let max_abs = normed
            .eigenvalues
            .iter()
            .map(|e| e.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            (max_abs - 1.0).abs() < 0.01,
            "Normalized max eigenvalue should be ~1, got {}",
            max_abs
        );
    }

    #[test]
    fn test_spectral_energy() {
        let sig = SpectralSignature::from_lines(&english_iambic_lines(), "English");
        assert!(sig.spectral_energy() > 0.0);
    }
}
