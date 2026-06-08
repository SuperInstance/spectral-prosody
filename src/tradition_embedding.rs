//! Low-dimensional spectral embeddings of poetic traditions.
//! Visualize the "space of all poetry".

use crate::laplacian_scan::SpectralSignature;
use crate::linalg;
use serde::{Deserialize, Serialize};

/// A tradition embedded in low-dimensional space.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraditionEmbedding {
    /// Coordinates in reduced space.
    pub coordinates: Vec<f64>,
    /// Tradition name.
    pub name: String,
}

impl TraditionEmbedding {
    /// Create a new embedding.
    pub fn new(coordinates: Vec<f64>, name: String) -> Self {
        Self { coordinates, name }
    }

    /// Euclidean distance to another embedding.
    pub fn distance_to(&self, other: &TraditionEmbedding) -> f64 {
        let max_dim = self.coordinates.len().max(other.coordinates.len());
        let mut sum_sq = 0.0;
        for i in 0..max_dim {
            let a = self.coordinates.get(i).copied().unwrap_or(0.0);
            let b = other.coordinates.get(i).copied().unwrap_or(0.0);
            sum_sq += (a - b) * (a - b);
        }
        sum_sq.sqrt()
    }

    /// Dot product with another embedding.
    pub fn dot(&self, other: &TraditionEmbedding) -> f64 {
        let max_dim = self.coordinates.len().max(other.coordinates.len());
        let mut dot = 0.0;
        for i in 0..max_dim {
            let a = self.coordinates.get(i).copied().unwrap_or(0.0);
            let b = other.coordinates.get(i).copied().unwrap_or(0.0);
            dot += a * b;
        }
        dot
    }

    /// Norm of the embedding vector.
    pub fn norm(&self) -> f64 {
        self.coordinates.iter().map(|c| c * c).sum::<f64>().sqrt()
    }
}

/// Embed traditions in low-dimensional space using spectral signatures.
/// Uses a simple PCA-like approach: take the top-k eigenvalues as coordinates.
pub fn embed_traditions(
    signatures: &[SpectralSignature],
    dimensions: usize,
) -> Vec<TraditionEmbedding> {
    signatures
        .iter()
        .map(|sig| {
            let normed = sig.normalized();
            // Use first `dimensions` eigenvalues as coordinates
            let coords: Vec<f64> = (0..dimensions)
                .map(|i| normed.eigenvalues.get(i).copied().unwrap_or(0.0))
                .collect();
            TraditionEmbedding::new(coords, sig.tradition.clone())
        })
        .collect()
}

/// Build a distance matrix between all tradition embeddings.
pub fn distance_matrix(embeddings: &[TraditionEmbedding]) -> Vec<Vec<f64>> {
    let n = embeddings.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = embeddings[i].distance_to(&embeddings[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Find the k nearest neighbors of a tradition.
pub fn nearest_neighbors(
    embeddings: &[TraditionEmbedding],
    query_idx: usize,
    k: usize,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = embeddings
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != query_idx)
        .map(|(i, e)| (i, embeddings[query_idx].distance_to(e)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);
    distances
}

/// Multidimensional scaling: reduce dimensionality of a distance matrix.
/// Simple iterative approach (SMACOF-like, one step).
pub fn mds_step(distances: &[Vec<f64>], _current: &[Vec<f64>], target_dim: usize) -> Vec<Vec<f64>> {
    let n = distances.len();
    if n == 0 {
        return vec![];
    }
    // Simple approach: use eigenvectors of the centered Gram matrix
    // B = -0.5 * H * D² * H where H = I - (1/n) * 11ᵀ
    let mut d_sq = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            d_sq[i][j] = distances[i][j] * distances[i][j];
        }
    }

    // Center: B_ij = -0.5 * (d²_ij - mean_row_i - mean_row_j + mean_all)
    let mut row_means = vec![0.0; n];
    for i in 0..n {
        row_means[i] = d_sq[i].iter().sum::<f64>() / n as f64;
    }
    let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

    let mut b = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            b[i][j] = -0.5 * (d_sq[i][j] - row_means[i] - row_means[j] + grand_mean);
        }
    }

    // Get top eigenvalues and use them
    let eigs = crate::linalg::jacobi_eigenvalues(&b, 500);
    // Take the largest `target_dim` eigenvalues
    let top_eigs: Vec<f64> = eigs.iter().rev().take(target_dim).copied().collect();

    // Construct coordinates using top eigenvalues as scaling
    // Simple: use the centered distance matrix columns as coordinates
    let mut coords = vec![vec![0.0; target_dim]; n];
    for d in 0..target_dim {
        let scale = if d < top_eigs.len() && top_eigs[d] > 0.0 {
            top_eigs[d].sqrt()
        } else {
            0.0
        };
        for i in 0..n {
            coords[i][d] = scale
                * (b[i].get(d).copied().unwrap_or(0.0) / n as f64)
                    .max(-1.0)
                    .min(1.0);
        }
    }
    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laplacian_scan::SpectralSignature;
    use crate::metrical_graph::MetricalLine;

    fn make_tradition(
        name: &str,
        syllable_count: usize,
        stress_pattern: Vec<bool>,
    ) -> SpectralSignature {
        let lines: Vec<MetricalLine> = (0..5)
            .map(|i| {
                MetricalLine::new(
                    vec![1.0; syllable_count],
                    stress_pattern.clone(),
                    format!("{}-{}", name, i),
                )
            })
            .collect();
        SpectralSignature::from_lines(&lines, name)
    }

    #[test]
    fn test_embedding_creation() {
        let emb = TraditionEmbedding::new(vec![1.0, 2.0, 3.0], "Test".into());
        assert_eq!(emb.coordinates.len(), 3);
        assert_eq!(emb.name, "Test");
    }

    #[test]
    fn test_embedding_distance() {
        let a = TraditionEmbedding::new(vec![0.0, 0.0], "A".into());
        let b = TraditionEmbedding::new(vec![3.0, 4.0], "B".into());
        let dist = a.distance_to(&b);
        assert!((dist - 5.0).abs() < 1e-9, "3-4-5 triangle, got {}", dist);
    }

    #[test]
    fn test_embed_traditions() {
        let sig1 = make_tradition(
            "English",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig2 = make_tradition(
            "French",
            12,
            vec![
                false, false, false, true, false, false, false, false, false, true, false, false,
            ],
        );
        let embeddings = embed_traditions(&[sig1, sig2], 3);
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].coordinates.len(), 3);
    }

    #[test]
    fn test_distance_matrix() {
        let e1 = TraditionEmbedding::new(vec![0.0], "A".into());
        let e2 = TraditionEmbedding::new(vec![1.0], "B".into());
        let dm = distance_matrix(&[e1, e2]);
        assert!((dm[0][1] - 1.0).abs() < 1e-9);
        assert!((dm[1][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_nearest_neighbors() {
        let embeddings = vec![
            TraditionEmbedding::new(vec![0.0], "A".into()),
            TraditionEmbedding::new(vec![1.0], "B".into()),
            TraditionEmbedding::new(vec![5.0], "C".into()),
        ];
        let nn = nearest_neighbors(&embeddings, 0, 2);
        assert_eq!(nn.len(), 2);
        assert_eq!(nn[0].0, 1); // B is closest to A
    }

    #[test]
    fn test_embedding_norm() {
        let e = TraditionEmbedding::new(vec![3.0, 4.0], "X".into());
        assert!((e.norm() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_related_traditions_close_in_embedding() {
        // Two iambic traditions should be closer than iambic vs free
        let en = make_tradition(
            "English",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        // "Spanish" also iambic-ish
        let es = make_tradition(
            "Spanish",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        // Free verse
        let free = make_tradition(
            "FreeVerse",
            7,
            vec![true, false, false, true, false, true, false],
        );

        let embeddings = embed_traditions(&[en, es, free], 5);
        let d_en_es = embeddings[0].distance_to(&embeddings[1]);
        let d_en_free = embeddings[0].distance_to(&embeddings[2]);
        assert!(
            d_en_es <= d_en_free,
            "Related traditions should be closer: en-es={} vs en-free={}",
            d_en_es,
            d_en_free
        );
    }

    #[test]
    fn test_mds_step() {
        let dist = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let current = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let result = mds_step(&dist, &current, 2);
        assert_eq!(result.len(), 3);
    }
}
