use crate::error::ProsodyError;
use crate::prosody::ProsodyGraph;
use crate::rhythm::RhythmExtractor;

/// A detected phrase within the prosody.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Phrase {
    pub start_idx: usize,
    pub end_idx: usize,
    pub label: String,
    pub average_energy: f64,
    pub contour: Vec<f64>,
}

/// Segments prosody into phrases using spectral clustering on the Fiedler vector.
pub struct PhraseSegmenter {
    /// Maximum number of phrases to extract.
    pub max_phrases: usize,
    /// Number of rhythm layers to use for clustering context.
    pub num_layers: usize,
}

impl Default for PhraseSegmenter {
    fn default() -> Self {
        Self {
            max_phrases: 8,
            num_layers: 5,
        }
    }
}

impl PhraseSegmenter {
    pub fn new(max_phrases: usize) -> Self {
        Self {
            max_phrases,
            num_layers: max_phrases * 2,
        }
    }

    /// Segment the graph into phrases using the Fiedler vector (2nd eigenvector).
    ///
    /// Strategy: Sort nodes by Fiedler vector value, then cut at sign changes
    /// or at the largest gaps in sorted values.
    pub fn segment(&self, graph: &ProsodyGraph) -> Result<Vec<Phrase>, ProsodyError> {
        let n = graph.nodes.len();
        if n < 2 {
            return Err(ProsodyError::InsufficientNodes { got: n, need: 2 });
        }

        let extractor = RhythmExtractor::new(self.num_layers);
        let layers = extractor.extract(graph)?;

        if layers.is_empty() {
            return Ok(vec![self.make_phrase(graph, 0, n - 1, "phrase_0")]);
        }

        // Use the Fiedler vector (2nd smallest eigenvalue's eigenvector) for bipartitioning
        let fiedler = if layers.len() >= 2 {
            &layers[1].eigenvector
        } else {
            &layers[0].eigenvector
        };

        // Create sorted index by Fiedler value
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| fiedler[a].partial_cmp(&fiedler[b]).unwrap());

        // Find natural boundaries: where Fiedler vector changes sign or has large gaps
        let mut boundaries = Vec::new();
        
        // Sign-change boundaries
        let mut prev_sign = fiedler[indices[0]].signum();
        for (k, &idx) in indices.iter().enumerate().skip(1) {
            let cur_sign = fiedler[idx].signum();
            if cur_sign != prev_sign && cur_sign != 0.0 {
                boundaries.push(k);
                prev_sign = cur_sign;
            }
        }

        // If we have too few boundaries, add gap-based boundaries
        if boundaries.len() < self.max_phrases.saturating_sub(1) {
            let mut gaps: Vec<(usize, f64)> = Vec::new();
            for k in 1..n {
                let gap = (fiedler[indices[k]] - fiedler[indices[k - 1]]).abs();
                gaps.push((k, gap));
            }
            gaps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut all_boundaries: std::collections::BTreeSet<usize> = boundaries.into_iter().collect();
            for (k, _) in gaps {
                if all_boundaries.len() >= self.max_phrases.saturating_sub(1) {
                    break;
                }
                all_boundaries.insert(k);
            }
            boundaries = all_boundaries.into_iter().collect();
        }

        // Truncate to max_phrases - 1 boundaries
        boundaries.truncate(self.max_phrases.saturating_sub(1));
        boundaries.sort_unstable();

        // Build phrases from boundaries (in original index order)
        let mut phrases = Vec::new();
        let mut start = 0;

        for &boundary in &boundaries {
            let end = boundary.saturating_sub(1);
            if end >= start {
                // Map back to original indices
                let phrase_indices: Vec<usize> = indices[start..=end].to_vec();
                let min_idx = *phrase_indices.iter().min().unwrap_or(&0);
                let max_idx = *phrase_indices.iter().max().unwrap_or(&0);
                let label = format!("phrase_{}", phrases.len());
                phrases.push(self.make_phrase_from_indices(graph, &phrase_indices, min_idx, max_idx, &label));
            }
            start = boundary;
        }

        // Final phrase
        if start < n {
            let phrase_indices: Vec<usize> = indices[start..].to_vec();
            let min_idx = *phrase_indices.iter().min().unwrap_or(&0);
            let max_idx = *phrase_indices.iter().max().unwrap_or(&(n - 1));
            let label = format!("phrase_{}", phrases.len());
            phrases.push(self.make_phrase_from_indices(graph, &phrase_indices, min_idx, max_idx, &label));
        }

        Ok(phrases)
    }

    fn make_phrase(&self, graph: &ProsodyGraph, start: usize, end: usize, label: &str) -> Phrase {
        let nodes = &graph.nodes[start..=end];
        let average_energy = nodes.iter().map(|n| n.energy).sum::<f64>() / nodes.len() as f64;
        let contour: Vec<f64> = nodes.iter().map(|n| n.pitch).collect();
        Phrase {
            start_idx: start,
            end_idx: end,
            label: label.to_string(),
            average_energy,
            contour,
        }
    }

    fn make_phrase_from_indices(
        &self,
        graph: &ProsodyGraph,
        indices: &[usize],
        min_idx: usize,
        max_idx: usize,
        label: &str,
    ) -> Phrase {
        let average_energy = indices
            .iter()
            .map(|&i| graph.nodes[i].energy)
            .sum::<f64>()
            / indices.len().max(1) as f64;
        let contour: Vec<f64> = indices
            .iter()
            .map(|&i| graph.nodes[i].pitch)
            .collect();
        Phrase {
            start_idx: min_idx,
            end_idx: max_idx,
            label: label.to_string(),
            average_energy,
            contour,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::ProsodyNode;

    fn make_graph(n: usize) -> ProsodyGraph {
        let nodes: Vec<ProsodyNode> = (0..n)
            .map(|i| ProsodyNode::new(i as f64 * 0.5, 1.0, 220.0 + i as f64 * 10.0, 0.25, 3000.0))
            .collect();
        ProsodyGraph::build_full(nodes, 1.0).unwrap()
    }

    fn make_two_cluster_graph() -> ProsodyGraph {
        let mut nodes = Vec::new();
        // Cluster 1: times 0-2
        for i in 0..6 {
            nodes.push(ProsodyNode::new(i as f64 * 0.4, 1.0, 200.0, 0.2, 2500.0));
        }
        // Cluster 2: times 10-12
        for i in 0..6 {
            nodes.push(ProsodyNode::new(10.0 + i as f64 * 0.4, 0.8, 350.0, 0.2, 4000.0));
        }
        ProsodyGraph::build_full(nodes, 1.0).unwrap()
    }

    #[test]
    fn test_segment_basic() {
        let graph = make_graph(10);
        let segmenter = PhraseSegmenter::new(4);
        let phrases = segmenter.segment(&graph).unwrap();
        assert!(!phrases.is_empty());
        assert!(phrases.len() <= 4);
        // All nodes should be covered
        let total_nodes: usize = phrases.iter().map(|p| p.end_idx - p.start_idx + 1).sum();
        assert!(total_nodes >= 10);
    }

    #[test]
    fn test_segment_two_clusters() {
        let graph = make_two_cluster_graph();
        let segmenter = PhraseSegmenter::new(4);
        let phrases = segmenter.segment(&graph).unwrap();
        assert!(phrases.len() >= 2, "should find at least 2 phrases for 2 clusters");
    }

    #[test]
    fn test_phrase_fields() {
        let graph = make_graph(5);
        let segmenter = PhraseSegmenter::new(2);
        let phrases = segmenter.segment(&graph).unwrap();
        for phrase in &phrases {
            assert!(!phrase.label.is_empty());
            assert!(phrase.average_energy >= 0.0);
            assert!(!phrase.contour.is_empty());
            assert!(phrase.start_idx <= phrase.end_idx);
        }
    }

    #[test]
    fn test_too_few_nodes() {
        let nodes = vec![ProsodyNode::new(0.0, 1.0, 220.0, 0.25, 3000.0)];
        let graph = ProsodyGraph {
            nodes,
            edges: vec![vec![0.0]],
            laplacian: vec![vec![0.0]],
        };
        let segmenter = PhraseSegmenter::new(4);
        assert!(segmenter.segment(&graph).is_err());
    }

    #[test]
    fn test_single_phrase_when_one_cluster() {
        // Very tightly clustered nodes → likely single phrase
        let nodes: Vec<ProsodyNode> = (0..4)
            .map(|i| ProsodyNode::new(i as f64 * 0.01, 1.0, 220.0, 0.01, 3000.0))
            .collect();
        let graph = ProsodyGraph::build_full(nodes, 10.0).unwrap();
        let segmenter = PhraseSegmenter::new(2);
        let phrases = segmenter.segment(&graph).unwrap();
        assert!(!phrases.is_empty());
    }

    #[test]
    fn test_contour_values() {
        let graph = make_graph(6);
        let segmenter = PhraseSegmenter::new(2);
        let phrases = segmenter.segment(&graph).unwrap();
        // Contour should contain pitch values
        for phrase in &phrases {
            for &val in &phrase.contour {
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn test_three_cluster_graph() {
        let mut nodes = Vec::new();
        for i in 0..4 {
            nodes.push(ProsodyNode::new(i as f64 * 0.3, 1.0, 200.0, 0.15, 2500.0));
        }
        for i in 0..4 {
            nodes.push(ProsodyNode::new(8.0 + i as f64 * 0.3, 0.8, 300.0, 0.15, 3500.0));
        }
        for i in 0..4 {
            nodes.push(ProsodyNode::new(16.0 + i as f64 * 0.3, 0.6, 400.0, 0.15, 4500.0));
        }
        let graph = ProsodyGraph::build_full(nodes, 1.0).unwrap();
        let segmenter = PhraseSegmenter::new(5);
        let phrases = segmenter.segment(&graph).unwrap();
        assert!(phrases.len() >= 2);
    }
}
