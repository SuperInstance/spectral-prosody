//! Connect to dial-theory: spectral distance between traditions as one dial dimension.
//! A "dial" is a dimension along which poetic traditions vary.

use crate::laplacian_scan::SpectralSignature;
use crate::tradition_embedding::TraditionEmbedding;
use serde::{Deserialize, Serialize};

/// A dial dimension measuring variation across traditions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DialDimension {
    /// Name of this dimension.
    pub name: String,
    /// Position of each tradition along this dimension.
    pub positions: Vec<(String, f64)>,
}

impl DialDimension {
    /// Create a new dial dimension.
    pub fn new(name: &str, positions: Vec<(String, f64)>) -> Self {
        Self {
            name: name.to_string(),
            positions,
        }
    }

    /// Range of positions.
    pub fn range(&self) -> (f64, f64) {
        let vals: Vec<f64> = self.positions.iter().map(|(_, v)| *v).collect();
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// Distance between two traditions on this dial.
    pub fn dial_distance(&self, tradition_a: &str, tradition_b: &str) -> Option<f64> {
        let pos_a = self
            .positions
            .iter()
            .find(|(name, _)| name == tradition_a)
            .map(|(_, v)| *v)?;
        let pos_b = self
            .positions
            .iter()
            .find(|(name, _)| name == tradition_b)
            .map(|(_, v)| *v)?;
        Some((pos_a - pos_b).abs())
    }
}

/// A full dial-space representation of traditions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DialSpace {
    /// Multiple dial dimensions.
    pub dimensions: Vec<DialDimension>,
    /// Tradition names.
    pub traditions: Vec<String>,
}

impl DialSpace {
    /// Create dial space from spectral signatures.
    /// Each eigenvalue index becomes one dial dimension.
    pub fn from_signatures(signatures: &[SpectralSignature], max_dims: usize) -> Self {
        if signatures.is_empty() {
            return Self {
                dimensions: vec![],
                traditions: vec![],
            };
        }

        let traditions: Vec<String> = signatures.iter().map(|s| s.tradition.clone()).collect();

        // Find max eigenvalue count
        let max_eigs = signatures
            .iter()
            .map(|s| s.eigenvalues.len())
            .max()
            .unwrap_or(0);
        let dim_count = max_eigs.min(max_dims);

        let mut dimensions = Vec::new();
        for d in 0..dim_count {
            let positions: Vec<(String, f64)> = signatures
                .iter()
                .map(|sig| {
                    let val = sig.normalized().eigenvalues.get(d).copied().unwrap_or(0.0);
                    (sig.tradition.clone(), val)
                })
                .collect();
            dimensions.push(DialDimension::new(&format!("λ_{}", d), positions));
        }

        Self {
            dimensions,
            traditions,
        }
    }

    /// Full dial distance between two traditions (Euclidean across all dimensions).
    pub fn full_dial_distance(&self, a: &str, b: &str) -> f64 {
        let mut sum_sq = 0.0;
        for dim in &self.dimensions {
            if let Some(d) = dim.dial_distance(a, b) {
                sum_sq += d * d;
            }
        }
        sum_sq.sqrt()
    }

    /// Number of dimensions.
    pub fn dimension_count(&self) -> usize {
        self.dimensions.len()
    }
}

/// Compute dial distance directly between two spectral signatures.
pub fn spectral_dial_distance(a: &SpectralSignature, b: &SpectralSignature) -> f64 {
    a.normalized().distance_to(&b.normalized())
}

/// Rank traditions by proximity to a query tradition in dial space.
pub fn rank_by_proximity(space: &DialSpace, query: &str) -> Vec<(String, f64)> {
    let mut results: Vec<(String, f64)> = space
        .traditions
        .iter()
        .filter(|t| *t != query)
        .map(|t| (t.clone(), space.full_dial_distance(query, t)))
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrical_graph::MetricalLine;

    fn make_signature(name: &str, n_syl: usize, stresses: Vec<bool>) -> SpectralSignature {
        let lines: Vec<MetricalLine> = (0..4)
            .map(|i| {
                MetricalLine::new(
                    vec![1.0; n_syl],
                    stresses.clone(),
                    format!("{}-{}", name, i),
                )
            })
            .collect();
        SpectralSignature::from_lines(&lines, name)
    }

    #[test]
    fn test_dial_dimension_creation() {
        let dim = DialDimension::new(
            "rhythmic_density",
            vec![
                ("English".into(), 0.5),
                ("French".into(), 0.3),
                ("Sanskrit".into(), 0.7),
            ],
        );
        assert_eq!(dim.positions.len(), 3);
        let (min, max) = dim.range();
        assert!((min - 0.3).abs() < 1e-9);
        assert!((max - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_dial_distance() {
        let dim = DialDimension::new("test", vec![("A".into(), 1.0), ("B".into(), 4.0)]);
        assert!((dim.dial_distance("A", "B").unwrap() - 3.0).abs() < 1e-9);
        assert!(dim.dial_distance("A", "C").is_none());
    }

    #[test]
    fn test_dial_space_from_signatures() {
        let sig1 = make_signature(
            "English",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig2 = make_signature(
            "French",
            12,
            vec![
                false, false, false, true, false, false, false, false, false, true, false, false,
            ],
        );
        let space = DialSpace::from_signatures(&[sig1, sig2], 3);
        assert_eq!(space.traditions.len(), 2);
        assert!(space.dimension_count() > 0);
    }

    #[test]
    fn test_full_dial_distance() {
        let sig1 = make_signature(
            "A",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig2 = make_signature(
            "B",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig3 = make_signature("C", 7, vec![true, false, false, true, false, true, false]);
        let space = DialSpace::from_signatures(&[sig1, sig2, sig3], 5);
        let d_ab = space.full_dial_distance("A", "B");
        let d_ac = space.full_dial_distance("A", "C");
        assert!(
            d_ab <= d_ac,
            "Same-meter traditions should be closer: ab={} vs ac={}",
            d_ab,
            d_ac
        );
    }

    #[test]
    fn test_spectral_dial_distance() {
        let sig1 = make_signature(
            "A",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig2 = make_signature(
            "A",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let dist = spectral_dial_distance(&sig1, &sig2);
        assert!(
            dist < 0.1,
            "Same tradition should have near-zero dial distance, got {}",
            dist
        );
    }

    #[test]
    fn test_rank_by_proximity() {
        let sig1 = make_signature(
            "English",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig2 = make_signature(
            "Spanish",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let sig3 = make_signature(
            "Free",
            7,
            vec![true, false, false, true, false, true, false],
        );
        let space = DialSpace::from_signatures(&[sig1, sig2, sig3], 5);
        let ranked = rank_by_proximity(&space, "English");
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, "Spanish"); // Closest
    }

    #[test]
    fn test_dial_space_serde() {
        let sig = make_signature(
            "Test",
            10,
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
        );
        let space = DialSpace::from_signatures(&[sig], 3);
        let json = serde_json::to_string(&space).unwrap();
        let back: DialSpace = serde_json::from_str(&json).unwrap();
        assert_eq!(back.traditions, space.traditions);
    }
}
