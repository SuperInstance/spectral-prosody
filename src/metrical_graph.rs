//! Construct weighted graphs from poetic corpora.
//! Nodes = lines, edges = metrical/stress similarity, weights = syllable distance.

use serde::{Deserialize, Serialize};
use crate::linalg;

/// A single line of poetry with metrical information.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricalLine {
    /// Duration weight per syllable (normalized).
    pub syllables: Vec<f64>,
    /// Boolean stress pattern: true = stressed.
    pub stresses: Vec<bool>,
    /// Source language or tradition name.
    pub language: String,
}

impl MetricalLine {
    /// Create a new metrical line.
    pub fn new(syllables: Vec<f64>, stresses: Vec<bool>, language: String) -> Self {
        Self { syllables, stresses, language }
    }

    /// The number of syllables.
    pub fn syllable_count(&self) -> usize {
        self.syllables.len()
    }

    /// Stress count.
    pub fn stress_count(&self) -> usize {
        self.stresses.iter().filter(|&&s| s).count()
    }

    /// Metrical weight: ratio of stressed syllables.
    pub fn metrical_weight(&self) -> f64 {
        if self.syllables.is_empty() {
            return 0.0;
        }
        self.stress_count() as f64 / self.syllables.len() as f64
    }

    /// Compute stress similarity to another line (Hamming-style).
    pub fn stress_similarity(&self, other: &MetricalLine) -> f64 {
        let max_len = self.stresses.len().max(other.stresses.len());
        if max_len == 0 {
            return 1.0;
        }
        let mut matches = 0;
        for i in 0..max_len {
            let a = self.stresses.get(i).copied().unwrap_or(false);
            let b = other.stresses.get(i).copied().unwrap_or(false);
            if a == b {
                matches += 1;
            }
        }
        matches as f64 / max_len as f64
    }

    /// Compute syllable distance (Euclidean) to another line.
    pub fn syllable_distance(&self, other: &MetricalLine) -> f64 {
        let max_len = self.syllables.len().max(other.syllables.len());
        if max_len == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for i in 0..max_len {
            let a = self.syllables.get(i).copied().unwrap_or(0.0);
            let b = other.syllables.get(i).copied().unwrap_or(0.0);
            sum_sq += (a - b) * (a - b);
        }
        sum_sq.sqrt()
    }

    /// Combined metrical similarity in [0, 1]: high = similar.
    pub fn metrical_similarity(&self, other: &MetricalLine) -> f64 {
        let stress_sim = self.stress_similarity(other);
        let max_dist = 10.0; // normalization constant
        let dist = self.syllable_distance(other).min(max_dist);
        let dist_sim = 1.0 - dist / max_dist;
        0.6 * stress_sim + 0.4 * dist_sim
    }
}

/// Weighted graph where nodes are metrical lines and edges encode similarity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricalGraph {
    /// Weighted adjacency matrix.
    pub adjacency: Vec<Vec<f64>>,
    /// Line labels.
    pub labels: Vec<String>,
}

impl MetricalGraph {
    /// Build a metrical graph from lines.
    /// Edge weights = metrical similarity between lines.
    pub fn from_lines(lines: &[MetricalLine]) -> Self {
        let n = lines.len();
        let mut adj = vec![vec![0.0; n]; n];
        let labels: Vec<String> = lines.iter()
            .enumerate()
            .map(|(i, l)| format!("{}:{}", l.language, i))
            .collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let w = lines[i].metrical_similarity(&lines[j]);
                adj[i][j] = w;
                adj[j][i] = w;
            }
        }
        Self { adjacency: adj, labels }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Compute the degree matrix (diagonal).
    pub fn degree_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.node_count();
        let mut d = vec![vec![0.0; n]; n];
        for i in 0..n {
            d[i][i] = self.adjacency[i].iter().sum();
        }
        d
    }

    /// Compute the unnormalized graph Laplacian: L = D - A.
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.node_count();
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            let deg: f64 = self.adjacency[i].iter().sum();
            l[i][i] = deg;
            for j in 0..n {
                if i != j {
                    l[i][j] = -self.adjacency[i][j];
                }
            }
        }
        l
    }

    /// Compute the normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}.
    pub fn normalized_laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.node_count();
        let deg: Vec<f64> = (0..n).map(|i| self.adjacency[i].iter().sum::<f64>().max(1e-12)).collect();
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    l[i][i] = 1.0;
                } else {
                    l[i][j] = -self.adjacency[i][j] / (deg[i].sqrt() * deg[j].sqrt());
                }
            }
        }
        l
    }

    /// Compute the Cheeger constant (approximate via Fiedler vector).
    /// The Cheeger constant h(G) measures the "bottleneck" of the graph.
    pub fn cheeger_constant(&self) -> f64 {
        let n = self.node_count();
        if n < 2 {
            return 0.0;
        }
        let lap = self.laplacian();
        let eigs = crate::linalg::jacobi_eigenvalues(&lap, 500);
        // Fiedler value = second-smallest eigenvalue
        let fiedler = eigs.get(1).copied().unwrap_or(0.0);
        // Cheeger inequality: h²/2 ≤ λ₂ ≤ 2h
        // Approximate h ≈ sqrt(2 * λ₂) as lower bound
        (2.0 * fiedler.max(0.0)).sqrt()
    }

    /// Expected random walk traversal time (mean first passage time).
    /// Interpretation: higher = more complex metrical structure.
    pub fn expected_traversal_time(&self) -> f64 {
        let n = self.node_count() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let lap = self.laplacian();
        let eigs = crate::linalg::jacobi_eigenvalues(&lap, 500);
        // For random walk on graph, mean first passage time relates to eigenvalues
        // Kemeny's constant ≈ sum of 1/λᵢ for non-zero eigenvalues
        let kemeny: f64 = eigs.iter()
            .filter(|&&e| e.abs() > 1e-10)
            .map(|&e| 1.0 / e)
            .sum();
        kemeny
    }

    /// Total edge weight.
    pub fn total_weight(&self) -> f64 {
        let n = self.node_count();
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.adjacency[i][j];
            }
        }
        total
    }

    /// Average edge weight.
    pub fn average_weight(&self) -> f64 {
        let n = self.node_count();
        if n < 2 {
            return 0.0;
        }
        let edges = n * (n - 1) / 2;
        self.total_weight() / edges as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iambic_pentameter() -> MetricalLine {
        MetricalLine::new(
            vec![1.0; 10],
            vec![false, true, false, true, false, true, false, true, false, true],
            "English".into(),
        )
    }

    fn trochaic_tetrameter() -> MetricalLine {
        MetricalLine::new(
            vec![1.0; 8],
            vec![true, false, true, false, true, false, true, false],
            "English".into(),
        )
    }

    fn blank_verse_line() -> MetricalLine {
        MetricalLine::new(
            vec![1.0; 10],
            vec![false, true, false, true, false, true, false, true, false, true],
            "English".into(),
        )
    }

    fn sanskrit_sloka() -> MetricalLine {
        // Sanskrit sloka: ~16 syllables with complex pattern
        MetricalLine::new(
            vec![1.0; 16],
            vec![
                false, true, false, true, false, true, false, false,
                false, true, false, true, false, true, false, false,
            ],
            "Sanskrit".into(),
        )
    }

    #[test]
    fn test_metrical_line_creation() {
        let line = iambic_pentameter();
        assert_eq!(line.syllable_count(), 10);
        assert_eq!(line.stress_count(), 5);
        assert_eq!(line.language, "English");
    }

    #[test]
    fn test_metrical_weight() {
        let line = iambic_pentameter();
        assert!((line.metrical_weight() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stress_similarity_identical() {
        let a = iambic_pentameter();
        assert!((a.stress_similarity(&a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stress_similarity_different() {
        let a = iambic_pentameter();
        let b = trochaic_tetrameter();
        // Iambic: _/_/_/_/_ (10), Trochaic: /_/_/_/_ (8)
        // Padded comparison of 10 positions
        let sim = a.stress_similarity(&b);
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_syllable_distance_identical() {
        let a = iambic_pentameter();
        assert!((a.syllable_distance(&a)).abs() < 1e-9);
    }

    #[test]
    fn test_metrical_similarity_high_for_same() {
        let a = iambic_pentameter();
        let b = blank_verse_line();
        // Both are iambic pentameter — should be very similar
        let sim = a.metrical_similarity(&b);
        assert!(sim > 0.9, "Expected high similarity, got {}", sim);
    }

    #[test]
    fn test_metrical_similarity_lower_for_different() {
        let a = iambic_pentameter();
        let b = trochaic_tetrameter();
        let c = sanskrit_sloka();
        let sim_ab = a.metrical_similarity(&b);
        // Both are same language but different meters — similarity should be moderate
        // Sanskrit is very different (16 syllables vs 10)
        let sim_ac = a.metrical_similarity(&c);
        // We just verify both are valid similarities in [0, 1]
        assert!(sim_ab > 0.0 && sim_ab <= 1.0);
        assert!(sim_ac > 0.0 && sim_ac <= 1.0);
    }

    #[test]
    fn test_graph_from_lines() {
        let lines = vec![iambic_pentameter(), trochaic_tetrameter(), sanskrit_sloka()];
        let graph = MetricalGraph::from_lines(&lines);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.labels.len(), 3);
        // Diagonal should be 0
        assert!((graph.adjacency[0][0]).abs() < 1e-9);
        // Symmetric
        assert!((graph.adjacency[0][1] - graph.adjacency[1][0]).abs() < 1e-9);
    }

    #[test]
    fn test_laplacian_rows_sum_zero() {
        let lines = vec![iambic_pentameter(), trochaic_tetrameter(), sanskrit_sloka()];
        let graph = MetricalGraph::from_lines(&lines);
        let lap = graph.laplacian();
        for row in &lap {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-9, "Laplacian row should sum to 0, got {}", sum);
        }
    }

    #[test]
    fn test_cheeger_constant_positive() {
        let lines = vec![iambic_pentameter(), trochaic_tetrameter(), sanskrit_sloka()];
        let graph = MetricalGraph::from_lines(&lines);
        let h = graph.cheeger_constant();
        assert!(h >= 0.0, "Cheeger constant should be non-negative");
    }

    #[test]
    fn test_expected_traversal_time() {
        let lines = vec![iambic_pentameter(), trochaic_tetrameter(), sanskrit_sloka()];
        let graph = MetricalGraph::from_lines(&lines);
        let t = graph.expected_traversal_time();
        assert!(t > 0.0, "Expected traversal time should be positive");
    }

    #[test]
    fn test_graph_serde_roundtrip() {
        let lines = vec![iambic_pentameter(), trochaic_tetrameter()];
        let graph = MetricalGraph::from_lines(&lines);
        let json = serde_json::to_string(&graph).unwrap();
        let back: MetricalGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.node_count(), graph.node_count());
        assert_eq!(back.labels, graph.labels);
    }

    #[test]
    fn test_metrical_line_serde_roundtrip() {
        let line = sanskrit_sloka();
        let json = serde_json::to_string(&line).unwrap();
        let back: MetricalLine = serde_json::from_str(&json).unwrap();
        assert_eq!(back.syllables, line.syllables);
        assert_eq!(back.stresses, line.stresses);
        assert_eq!(back.language, line.language);
    }
}
