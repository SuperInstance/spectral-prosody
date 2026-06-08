use crate::error::ProsodyError;
use crate::prosody::ProsodyGraph;

/// A single rhythmic layer extracted from the graph Laplacian spectrum.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RhythmLayer {
    pub eigenvalue: f64,
    pub eigenvector: Vec<f64>,
    pub period: f64,
    pub strength: f64,
}

/// Extracts rhythmic layers via spectral decomposition of the graph Laplacian.
pub struct RhythmExtractor {
    /// Maximum number of layers to extract.
    pub max_layers: usize,
    /// Convergence tolerance for power iteration.
    pub tolerance: f64,
    /// Maximum iterations per eigenpair.
    pub max_iterations: usize,
}

impl Default for RhythmExtractor {
    fn default() -> Self {
        Self {
            max_layers: 10,
            tolerance: 1e-8,
            max_iterations: 1000,
        }
    }
}

impl RhythmExtractor {
    pub fn new(max_layers: usize) -> Self {
        Self {
            max_layers,
            ..Default::default()
        }
    }

    /// Extract rhythmic layers from the graph Laplacian using power iteration + deflation.
    ///
    /// Low eigenvalues → macro rhythm (long periods).
    /// High eigenvalues → micro rhythm (short periods).
    pub fn extract(&self, graph: &ProsodyGraph) -> Result<Vec<RhythmLayer>, ProsodyError> {
        let n = graph.nodes.len();
        if n < 2 {
            return Err(ProsodyError::InsufficientNodes { got: n, need: 2 });
        }

        let num_layers = self.max_layers.min(n);
        let mut layers = Vec::with_capacity(num_layers);
        let mut deflated = graph.laplacian.clone();

        for k in 0..num_layers {
            let (eigenvalue, eigenvector) = self.power_iteration(&deflated, n)?;

            // Deflate: remove this eigenpair contribution
            for i in 0..n {
                for j in 0..n {
                    deflated[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
                }
            }

            // Map eigenvalue to rhythmic period.
            // Using total time span as reference.
            let total_time = graph.nodes.last().unwrap().time - graph.nodes.first().unwrap().time;
            let period = if eigenvalue.abs() > 1e-12 {
                2.0 * std::f64::consts::PI / eigenvalue.sqrt().max(1e-12)
            } else {
                total_time * 2.0 // near-zero eigenvalue → very long period
            };

            // Strength proportional to 1/eigenvalue (lower eigenvalues = stronger structure)
            let max_eigenvalue = eigenvalue.max(1e-12);
            let strength = 1.0 / (1.0 + max_eigenvalue);

            layers.push(RhythmLayer {
                eigenvalue,
                eigenvector,
                period: period.min(total_time * 4.0),
                strength,
            });

            // Stop if eigenvalue is essentially zero (disconnected components exhausted)
            if eigenvalue.abs() < self.tolerance && k > 0 {
                break;
            }
        }

        // Sort by eigenvalue ascending
        layers.sort_by(|a, b| a.eigenvalue.partial_cmp(&b.eigenvalue).unwrap());

        Ok(layers)
    }

    /// Power iteration to find the dominant eigenvector of a matrix.
    fn power_iteration(&self, matrix: &[Vec<f64>], n: usize) -> Result<(f64, Vec<f64>), ProsodyError> {
        // Start with a random-ish initial vector (deterministic seed via index)
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.01).collect();
        Self::normalize(&mut v);

        let mut eigenvalue = 0.0_f64;

        for _ in 0..self.max_iterations {
            // Matrix-vector multiply
            let mut mv = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    mv[i] += matrix[i][j] * v[j];
                }
            }

            // Rayleigh quotient for eigenvalue estimate
            let new_eigenvalue: f64 = mv.iter().zip(&v).map(|(m, vi)| m * vi).sum();

            Self::normalize(&mut mv);

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < self.tolerance {
                eigenvalue = new_eigenvalue;
                v = mv;
                break;
            }
            eigenvalue = new_eigenvalue;
            v = mv;
        }

        // Ensure eigenvalue is non-negative (Laplacian is PSD)
        eigenvalue = eigenvalue.max(0.0);

        Ok((eigenvalue, v))
    }

    fn normalize(v: &mut [f64]) {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// Compute the algebraic connectivity (Fiedler value) — the second smallest eigenvalue.
pub fn algebraic_connectivity(graph: &ProsodyGraph) -> Result<f64, ProsodyError> {
    let extractor = RhythmExtractor::new(2);
    let layers = extractor.extract(graph)?;
    if layers.len() >= 2 {
        Ok(layers[1].eigenvalue)
    } else if layers.len() == 1 {
        Ok(layers[0].eigenvalue)
    } else {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::ProsodyNode;

    fn make_graph(n: usize, sigma: f64) -> ProsodyGraph {
        let nodes: Vec<ProsodyNode> = (0..n)
            .map(|i| ProsodyNode::new(i as f64 * 0.5, 1.0, 220.0, 0.25, 3000.0))
            .collect();
        ProsodyGraph::build_full(nodes, sigma).unwrap()
    }

    #[test]
    fn test_extract_basic() {
        let graph = make_graph(10, 1.0);
        let extractor = RhythmExtractor::new(5);
        let layers = extractor.extract(&graph).unwrap();
        assert!(!layers.is_empty());
        assert!(layers.len() <= 5);
    }

    #[test]
    fn test_layers_sorted_by_eigenvalue() {
        let graph = make_graph(8, 0.5);
        let extractor = RhythmExtractor::new(4);
        let layers = extractor.extract(&graph).unwrap();
        for w in layers.windows(2) {
            assert!(w[0].eigenvalue <= w[1].eigenvalue + 1e-6);
        }
    }

    #[test]
    fn test_eigenvector_norm() {
        let graph = make_graph(6, 1.0);
        let extractor = RhythmExtractor::new(3);
        let layers = extractor.extract(&graph).unwrap();
        for layer in &layers {
            let norm: f64 = layer.eigenvector.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "norm = {norm}");
        }
    }

    #[test]
    fn test_period_and_strength() {
        let graph = make_graph(10, 1.0);
        let extractor = RhythmExtractor::new(3);
        let layers = extractor.extract(&graph).unwrap();
        for layer in &layers {
            assert!(layer.period > 0.0);
            assert!(layer.strength > 0.0);
            assert!(layer.strength <= 1.0);
        }
    }

    #[test]
    fn test_algebraic_connectivity() {
        let graph = make_graph(8, 1.0);
        let ac = algebraic_connectivity(&graph).unwrap();
        assert!(ac >= 0.0);
    }

    #[test]
    fn test_too_few_nodes() {
        let nodes = vec![ProsodyNode::new(0.0, 1.0, 220.0, 0.25, 3000.0)];
        let graph = ProsodyGraph {
            nodes,
            edges: vec![vec![0.0]],
            laplacian: vec![vec![0.0]],
        };
        let extractor = RhythmExtractor::new(5);
        assert!(extractor.extract(&graph).is_err());
    }

    #[test]
    fn test_eigenvalue_nonnegative() {
        let graph = make_graph(12, 0.8);
        let extractor = RhythmExtractor::new(6);
        let layers = extractor.extract(&graph).unwrap();
        for layer in &layers {
            assert!(layer.eigenvalue >= 0.0);
        }
    }

    #[test]
    fn test_extract_two_cluster_rhythm() {
        // Two clusters of nodes far apart → should show up in spectral structure
        let mut nodes = Vec::new();
        for i in 0..5 {
            nodes.push(ProsodyNode::new(i as f64 * 0.3, 1.0, 220.0, 0.15, 3000.0));
        }
        for i in 0..5 {
            nodes.push(ProsodyNode::new(10.0 + i as f64 * 0.3, 1.0, 220.0, 0.15, 3000.0));
        }
        let graph = ProsodyGraph::build_full(nodes, 1.0).unwrap();
        let extractor = RhythmExtractor::new(5);
        let layers = extractor.extract(&graph).unwrap();
        assert!(layers.len() >= 2);
        // First eigenvalue should be very small (nearly disconnected)
        assert!(layers[0].eigenvalue < 1.0);
    }
}
