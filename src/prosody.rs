use crate::error::ProsodyError;
use crate::feature::ProsodyFeature;

/// A single node in the prosody graph (beat / syllable / onset).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProsodyNode {
    pub time: f64,
    pub energy: f64,
    pub pitch: f64,
    pub duration: f64,
    pub spectral_centroid: f64,
}

/// Weighted graph built from prosody features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProsodyGraph {
    pub nodes: Vec<ProsodyNode>,
    pub edges: Vec<Vec<f64>>,
    pub laplacian: Vec<Vec<f64>>,
}

impl ProsodyNode {
    pub fn new(time: f64, energy: f64, pitch: f64, duration: f64, spectral_centroid: f64) -> Self {
        Self { time, energy, pitch, duration, spectral_centroid }
    }
}

impl ProsodyGraph {
    /// Build a k-nearest-neighbor graph from a set of prosody nodes.
    ///
    /// Edges are weighted using a Gaussian kernel on temporal proximity:
    ///   w(i,j) = exp(-||t_i - t_j||^2 / (2 * sigma^2))
    pub fn build_knn(nodes: Vec<ProsodyNode>, k: usize, sigma: f64) -> Result<Self, ProsodyError> {
        let n = nodes.len();
        if n < 2 {
            return Err(ProsodyError::InsufficientNodes { got: n, need: 2 });
        }
        if sigma <= 0.0 {
            return Err(ProsodyError::InvalidParameter("sigma must be positive".into()));
        }
        let k = k.min(n - 1).max(1);

        let mut edges = vec![vec![0.0_f64; n]; n];

        for i in 0..n {
            // Compute distances to all other nodes (temporal proximity)
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dt = nodes[i].time - nodes[j].time;
                    let de = nodes[i].energy - nodes[j].energy;
                    let dp = nodes[i].pitch - nodes[j].pitch;
                    (j, (dt * dt + de * de * 0.01 + dp * dp * 0.0001).sqrt())
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(j, _d) in dists.iter().take(k) {
                let dt = nodes[i].time - nodes[j].time;
                let w = (-dt * dt / (2.0 * sigma * sigma)).exp();
                edges[i][j] = w;
                edges[j][i] = w;
            }
        }

        let laplacian = Self::compute_laplacian(&edges);
        Ok(Self { nodes, edges, laplacian })
    }

    /// Build a fully connected graph with Gaussian kernel weighting.
    pub fn build_full(nodes: Vec<ProsodyNode>, sigma: f64) -> Result<Self, ProsodyError> {
        let n = nodes.len();
        if n < 2 {
            return Err(ProsodyError::InsufficientNodes { got: n, need: 2 });
        }
        if sigma <= 0.0 {
            return Err(ProsodyError::InvalidParameter("sigma must be positive".into()));
        }

        let mut edges = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let dt = nodes[i].time - nodes[j].time;
                let w = (-dt * dt / (2.0 * sigma * sigma)).exp();
                edges[i][j] = w;
                edges[j][i] = w;
            }
        }

        let laplacian = Self::compute_laplacian(&edges);
        Ok(Self { nodes, edges, laplacian })
    }

    /// Compute the unnormalized graph Laplacian: L = D - W.
    fn compute_laplacian(edges: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = edges.len();
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = edges[i].iter().sum();
            lap[i][i] = degree;
            for j in 0..n {
                if i != j {
                    lap[i][j] = -edges[i][j];
                }
            }
        }
        lap
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Total edge weight.
    pub fn total_edge_weight(&self) -> f64 {
        let n = self.edges.len();
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.edges[i][j];
            }
        }
        total
    }

    /// Degree of node i.
    pub fn degree(&self, i: usize) -> f64 {
        self.edges[i].iter().sum()
    }
}

/// Build a ProsodyGraph from feature vectors.
pub fn graph_from_features(
    times: ProsodyFeature,
    energies: ProsodyFeature,
    pitches: ProsodyFeature,
    durations: ProsodyFeature,
    centroids: ProsodyFeature,
    k: usize,
    sigma: f64,
) -> Result<ProsodyGraph, ProsodyError> {
    let n = times.values.len();
    if n < 2 {
        return Err(ProsodyError::InsufficientNodes { got: n, need: 2 });
    }
    if energies.values.len() != n || pitches.values.len() != n || durations.values.len() != n || centroids.values.len() != n {
        return Err(ProsodyError::InvalidParameter("all feature vectors must have the same length".into()));
    }

    let nodes: Vec<ProsodyNode> = (0..n)
        .map(|i| ProsodyNode {
            time: times.values[i],
            energy: energies.values[i],
            pitch: pitches.values[i],
            duration: durations.values[i],
            spectral_centroid: centroids.values[i],
        })
        .collect();

    ProsodyGraph::build_knn(nodes, k, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_nodes(n: usize) -> Vec<ProsodyNode> {
        (0..n)
            .map(|i| ProsodyNode::new(i as f64 * 0.5, 1.0, 220.0, 0.25, 3000.0))
            .collect()
    }

    #[test]
    fn test_build_knn_basic() {
        let nodes = make_uniform_nodes(10);
        let graph = ProsodyGraph::build_knn(nodes, 3, 1.0).unwrap();
        assert_eq!(graph.len(), 10);
        assert_eq!(graph.edges.len(), 10);
        assert_eq!(graph.laplacian.len(), 10);
        // Laplacian rows should sum to 0
        for i in 0..10 {
            let row_sum: f64 = graph.laplacian[i].iter().sum();
            assert!(row_sum.abs() < 1e-10, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_build_full_basic() {
        let nodes = make_uniform_nodes(5);
        let graph = ProsodyGraph::build_full(nodes, 1.0).unwrap();
        assert_eq!(graph.len(), 5);
        // Fully connected: all off-diagonal edges > 0
        assert!(graph.edges[0][1] > 0.0);
        assert!(graph.edges[0][4] > 0.0);
        assert!(graph.total_edge_weight() > 0.0);
    }

    #[test]
    fn test_insufficient_nodes() {
        let nodes = vec![ProsodyNode::new(0.0, 1.0, 220.0, 0.25, 3000.0)];
        assert!(ProsodyGraph::build_knn(nodes, 3, 1.0).is_err());
    }

    #[test]
    fn test_invalid_sigma() {
        let nodes = make_uniform_nodes(5);
        assert!(ProsodyGraph::build_knn(nodes.clone(), 3, 0.0).is_err());
        assert!(ProsodyGraph::build_full(nodes, -1.0).is_err());
    }

    #[test]
    fn test_symmetric_edges() {
        let nodes = make_uniform_nodes(8);
        let graph = ProsodyGraph::build_knn(nodes, 3, 0.5).unwrap();
        for i in 0..8 {
            for j in 0..8 {
                assert!((graph.edges[i][j] - graph.edges[j][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_degree_and_edge_weight() {
        let nodes = make_uniform_nodes(5);
        let graph = ProsodyGraph::build_full(nodes, 1.0).unwrap();
        for i in 0..5 {
            assert!(graph.degree(i) > 0.0);
        }
        assert!(graph.total_edge_weight() > 0.0);
    }

    #[test]
    fn test_graph_from_features() {
        let n = 6;
        let times = ProsodyFeature::new("time", (0..n).map(|i| i as f64 * 0.5).collect(), 1.0);
        let energies = ProsodyFeature::new("energy", vec![1.0; n], 1.0);
        let pitches = ProsodyFeature::new("pitch", vec![220.0; n], 1.0);
        let durations = ProsodyFeature::new("dur", vec![0.25; n], 1.0);
        let centroids = ProsodyFeature::new("centroid", vec![3000.0; n], 1.0);

        let graph = graph_from_features(times, energies, pitches, durations, centroids, 3, 1.0).unwrap();
        assert_eq!(graph.len(), 6);
    }

    #[test]
    fn test_graph_from_features_mismatched_lengths() {
        let times = ProsodyFeature::new("time", vec![0.0, 0.5, 1.0], 1.0);
        let energies = ProsodyFeature::new("energy", vec![1.0, 1.0], 1.0);
        let pitches = ProsodyFeature::new("pitch", vec![220.0; 3], 1.0);
        let durations = ProsodyFeature::new("dur", vec![0.25; 3], 1.0);
        let centroids = ProsodyFeature::new("centroid", vec![3000.0; 3], 1.0);

        assert!(graph_from_features(times, energies, pitches, durations, centroids, 3, 1.0).is_err());
    }

    #[test]
    fn test_empty_graph() {
        let graph = ProsodyGraph {
            nodes: vec![],
            edges: vec![],
            laplacian: vec![],
        };
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
    }
}
