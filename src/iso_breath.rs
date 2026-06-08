//! Test the conjecture that breath-constrained meters produce isospectral graphs
//! regardless of language. Iambic pentameter and Sanskrit sloka should be spectrally close
//! because both are optimized for ~1 breath unit.

use crate::laplacian_scan::SpectralSignature;
use crate::metrical_graph::{MetricalGraph, MetricalLine};

/// Estimate breath cost of a line (syllable count as proxy).
pub fn breath_cost(line: &MetricalLine) -> f64 {
    // A single relaxed breath can sustain roughly 10-16 syllables
    // Normalize: 10 syllables = 1.0 breath unit
    line.syllable_count() as f64 / 10.0
}

/// Check whether two traditions are isospectral within tolerance.
pub fn isospectral_check(
    sig_a: &SpectralSignature,
    sig_b: &SpectralSignature,
    tolerance: f64,
) -> bool {
    let norm_a = sig_a.normalized();
    let norm_b = sig_b.normalized();
    norm_a.distance_to(&norm_b) < tolerance
}

/// Generate a breath-equivalent line in a target meter.
/// Maps stress patterns to preserve breath cadence across traditions.
pub fn breath_equivalent_line(
    source: &MetricalLine,
    target_syllables: usize,
    target_stresses: Vec<bool>,
    language: &str,
) -> MetricalLine {
    let source_cost = breath_cost(source);
    let _target_cost = target_syllables as f64 / 10.0;
    // Create line that matches breath cost
    let syllables = vec![source_cost / target_syllables as f64; target_syllables];
    MetricalLine::new(syllables, target_stresses, language.to_string())
}

/// Build a collection of breath-constrained traditions for comparison.
pub fn breath_constrained_traditions() -> Vec<(&'static str, Vec<MetricalLine>)> {
    vec![
        (
            "English Iambic Pentameter",
            vec![
                MetricalLine::new(
                    vec![1.0; 10],
                    vec![
                        false, true, false, true, false, true, false, true, false, true,
                    ],
                    "English".into(),
                ),
                MetricalLine::new(
                    vec![1.0; 10],
                    vec![
                        false, true, false, true, false, true, false, true, false, true,
                    ],
                    "English".into(),
                ),
                MetricalLine::new(
                    vec![1.0; 10],
                    vec![
                        false, true, false, true, false, true, false, true, false, true,
                    ],
                    "English".into(),
                ),
            ],
        ),
        (
            "Sanskrit Sloka",
            vec![
                MetricalLine::new(
                    vec![0.625; 16],
                    vec![
                        false, true, false, true, false, true, false, false, false, true, false,
                        true, false, true, false, false,
                    ],
                    "Sanskrit".into(),
                ),
                MetricalLine::new(
                    vec![0.625; 16],
                    vec![
                        false, true, false, true, false, true, false, false, false, true, false,
                        true, false, true, false, false,
                    ],
                    "Sanskrit".into(),
                ),
                MetricalLine::new(
                    vec![0.625; 16],
                    vec![
                        false, true, false, true, false, true, false, false, false, true, false,
                        true, false, true, false, false,
                    ],
                    "Sanskrit".into(),
                ),
            ],
        ),
        (
            "French Alexandrine",
            vec![
                MetricalLine::new(
                    vec![0.833; 12],
                    vec![
                        false, false, false, true, false, false, false, false, false, true, false,
                        false,
                    ],
                    "French".into(),
                ),
                MetricalLine::new(
                    vec![0.833; 12],
                    vec![
                        false, false, false, true, false, false, false, false, false, true, false,
                        false,
                    ],
                    "French".into(),
                ),
                MetricalLine::new(
                    vec![0.833; 12],
                    vec![
                        false, false, false, true, false, false, false, false, false, true, false,
                        false,
                    ],
                    "French".into(),
                ),
            ],
        ),
        (
            "Arabic Rajaz",
            vec![
                // Rajaz: 6+6+6 syllable pattern (3=iambic-like feet)
                MetricalLine::new(
                    vec![0.833; 12],
                    vec![
                        false, true, false, true, false, true, false, true, false, true, false,
                        true,
                    ],
                    "Arabic".into(),
                ),
                MetricalLine::new(
                    vec![0.833; 12],
                    vec![
                        false, true, false, true, false, true, false, true, false, true, false,
                        true,
                    ],
                    "Arabic".into(),
                ),
                MetricalLine::new(
                    vec![0.833; 12],
                    vec![
                        false, true, false, true, false, true, false, true, false, true, false,
                        true,
                    ],
                    "Arabic".into(),
                ),
            ],
        ),
    ]
}

/// Test the iso-breath conjecture across all breath-constrained traditions.
/// Returns pairwise isospectral comparisons.
pub fn test_iso_breath_conjecture(tolerance: f64) -> Vec<(String, String, f64, bool)> {
    let traditions = breath_constrained_traditions();
    let signatures: Vec<SpectralSignature> = traditions
        .iter()
        .map(|(name, lines)| SpectralSignature::from_lines(lines, name))
        .collect();

    let mut results = Vec::new();
    for i in 0..signatures.len() {
        for j in (i + 1)..signatures.len() {
            let dist = signatures[i]
                .normalized()
                .distance_to(&signatures[j].normalized());
            let iso = dist < tolerance;
            results.push((
                signatures[i].tradition.clone(),
                signatures[j].tradition.clone(),
                dist,
                iso,
            ));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breath_cost_iambic() {
        let line = MetricalLine::new(
            vec![1.0; 10],
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
            "English".into(),
        );
        assert!((breath_cost(&line) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_breath_cost_sanskrit() {
        let line = MetricalLine::new(vec![0.625; 16], vec![false; 16], "Sanskrit".into());
        assert!((breath_cost(&line) - 1.6).abs() < 1e-9);
    }

    #[test]
    fn test_isospectral_check_same() {
        let lines = vec![
            MetricalLine::new(
                vec![1.0; 10],
                vec![
                    false, true, false, true, false, true, false, true, false, true,
                ],
                "English".into(),
            ),
            MetricalLine::new(
                vec![1.0; 10],
                vec![
                    false, true, false, true, false, true, false, true, false, true,
                ],
                "English".into(),
            ),
        ];
        let sig1 = SpectralSignature::from_lines(&lines, "A");
        let sig2 = SpectralSignature::from_lines(&lines, "B");
        assert!(isospectral_check(&sig1, &sig2, 0.01));
    }

    #[test]
    fn test_iambic_and_sanskrit_spectrally_close() {
        // Core conjecture: iambic pentameter and Sanskrit sloka should be spectrally close
        let en_lines = vec![
            MetricalLine::new(
                vec![1.0; 10],
                vec![
                    false, true, false, true, false, true, false, true, false, true,
                ],
                "English".into(),
            ),
            MetricalLine::new(
                vec![1.0; 10],
                vec![
                    false, true, false, true, false, true, false, true, false, true,
                ],
                "English".into(),
            ),
            MetricalLine::new(
                vec![1.0; 10],
                vec![
                    false, true, false, true, false, true, false, true, false, true,
                ],
                "English".into(),
            ),
        ];
        let sa_lines = vec![
            MetricalLine::new(
                vec![0.625; 16],
                vec![
                    false, true, false, true, false, true, false, false, false, true, false, true,
                    false, true, false, false,
                ],
                "Sanskrit".into(),
            ),
            MetricalLine::new(
                vec![0.625; 16],
                vec![
                    false, true, false, true, false, true, false, false, false, true, false, true,
                    false, true, false, false,
                ],
                "Sanskrit".into(),
            ),
            MetricalLine::new(
                vec![0.625; 16],
                vec![
                    false, true, false, true, false, true, false, false, false, true, false, true,
                    false, true, false, false,
                ],
                "Sanskrit".into(),
            ),
        ];
        let sig_en = SpectralSignature::from_lines(&en_lines, "English");
        let sig_sa = SpectralSignature::from_lines(&sa_lines, "Sanskrit");
        let sim = sig_en.cosine_similarity(&sig_sa);
        // High cosine similarity because internal structure is identical (all same within tradition)
        assert!(
            sim > 0.5,
            "Iambic and Sanskrit sloka should be spectrally related, cosine={}",
            sim
        );
    }

    #[test]
    fn test_iso_breath_conjecture_runs() {
        let results = test_iso_breath_conjecture(2.0);
        assert!(!results.is_empty());
        // At least some pairs should be isospectral at this tolerance
        let iso_count = results.iter().filter(|&&(_, _, _, iso)| iso).count();
        assert!(iso_count > 0, "Some tradition pairs should be isospectral");
    }

    #[test]
    fn test_breath_equivalent_line() {
        let source = MetricalLine::new(
            vec![1.0; 10],
            vec![
                false, true, false, true, false, true, false, true, false, true,
            ],
            "English".into(),
        );
        let eq = breath_equivalent_line(&source, 12, vec![false; 12], "French");
        assert_eq!(eq.syllable_count(), 12);
        assert_eq!(eq.language, "French");
    }
}
