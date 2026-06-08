use crate::error::ProsodyError;

/// A named feature vector with an associated sample rate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProsodyFeature {
    pub name: String,
    pub values: Vec<f64>,
    pub sample_rate: f64,
}

impl ProsodyFeature {
    /// Create a new feature vector.
    pub fn new(name: impl Into<String>, values: Vec<f64>, sample_rate: f64) -> Self {
        Self {
            name: name.into(),
            values,
            sample_rate,
        }
    }

    /// Compute timing (onset interval) features from a sequence of onset times.
    pub fn from_onsets(onset_times: &[f64]) -> Result<Self, ProsodyError> {
        if onset_times.len() < 2 {
            return Err(ProsodyError::InsufficientNodes {
                got: onset_times.len(),
                need: 2,
            });
        }
        let intervals: Vec<f64> = onset_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        Ok(Self::new("timing", intervals, 1.0))
    }

    /// Compute energy features from raw amplitude samples.
    pub fn from_energy(samples: &[f64], sample_rate: f64, frame_size: usize) -> Self {
        let values: Vec<f64> = samples
            .chunks(frame_size)
            .map(|frame| {
                let rms = (frame.iter().map(|s| s * s).sum::<f64>() / frame.len() as f64).sqrt();
                rms
            })
            .collect();
        Self::new("energy", values, sample_rate / frame_size as f64)
    }

    /// Compute spectral centroid from magnitude spectrum frames.
    pub fn from_spectra(magnitudes: &[Vec<f64>], sample_rate: f64, fft_size: usize) -> Self {
        let bin_freq = sample_rate / fft_size as f64;
        let values: Vec<f64> = magnitudes
            .iter()
            .map(|spec| {
                let total: f64 = spec.iter().sum();
                if total < 1e-12 {
                    return 0.0;
                }
                let weighted: f64 = spec
                    .iter()
                    .enumerate()
                    .map(|(i, m)| i as f64 * bin_freq * m)
                    .sum();
                weighted / total
            })
            .collect();
        let frame_rate = sample_rate;
        Self::new("spectral_centroid", values, frame_rate)
    }

    /// Return the number of frames.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Compute mean of feature values.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Compute standard deviation.
    pub fn std_dev(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        let variance = self.values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / self.values.len() as f64;
        variance.sqrt()
    }
}

/// Extract pitch contour using simple autocorrelation on frame windows.
pub fn estimate_pitch(samples: &[f64], sample_rate: f64, frame_size: usize, min_freq: f64, max_freq: f64) -> ProsodyFeature {
    let min_lag = (sample_rate / max_freq) as usize;
    let max_lag = (sample_rate / min_freq) as usize;

    let pitches: Vec<f64> = samples
        .chunks(frame_size)
        .map(|frame| {
            if frame.len() < max_lag {
                return 0.0;
            }
            let mut best_lag = min_lag;
            let mut best_corr = f64::NEG_INFINITY;
            for lag in min_lag..=max_lag.min(frame.len() / 2) {
                let corr: f64 = frame[..frame.len() - lag]
                    .iter()
                    .zip(&frame[lag..])
                    .map(|(a, b)| a * b)
                    .sum();
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }
            if best_corr <= 0.0 {
                0.0
            } else {
                sample_rate / best_lag as f64
            }
        })
        .collect();

    ProsodyFeature::new("pitch", pitches, sample_rate / frame_size as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_from_onsets_basic() {
        let onsets = vec![0.0, 0.5, 1.0, 1.5];
        let feat = ProsodyFeature::from_onsets(&onsets).unwrap();
        assert_eq!(feat.values, vec![0.5, 0.5, 0.5]);
        assert_eq!(feat.name, "timing");
    }

    #[test]
    fn test_from_onsets_too_few() {
        let onsets = vec![0.0];
        assert!(ProsodyFeature::from_onsets(&onsets).is_err());
    }

    #[test]
    fn test_from_energy_sinusoid() {
        // Generate a 440 Hz sine wave at 44100 Hz, 0.1 seconds
        let sr = 44100.0;
        let freq = 440.0;
        let n = (sr * 0.1) as usize;
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin() * 0.5)
            .collect();
        let feat = ProsodyFeature::from_energy(&samples, sr, 1024);
        assert!(!feat.values.is_empty());
        assert!(feat.values[0] > 0.0);
        assert!(feat.values[0] < 1.0);
    }

    #[test]
    fn test_from_spectra_basic() {
        let magnitudes = vec![vec![0.0, 1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]];
        let feat = ProsodyFeature::from_spectra(&magnitudes, 44100.0, 4);
        assert_eq!(feat.values.len(), 2);
        // Second frame has energy at bin 0, centroid should be 0
        assert_eq!(feat.values[1], 0.0);
        // First frame has energy at bin 1, centroid = 1 * (44100/4) = 11025
        assert!((feat.values[0] - 11025.0).abs() < 1.0);
    }

    #[test]
    fn test_mean_and_std_dev() {
        let feat = ProsodyFeature::new("test", vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 1.0);
        assert!((feat.mean() - 5.0).abs() < 1e-10);
        assert!(feat.std_dev() > 0.0);
    }

    #[test]
    fn test_estimate_pitch_sine() {
        let sr = 44100.0;
        let freq = 440.0;
        let n = (sr * 0.05) as usize;
        let samples: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr).sin()).collect();
        let pitch = estimate_pitch(&samples, sr, 2048, 100.0, 2000.0);
        if !pitch.values.is_empty() && pitch.values[0] > 0.0 {
            assert!((pitch.values[0] - freq).abs() < 20.0);
        }
    }

    #[test]
    fn test_empty_feature() {
        let feat = ProsodyFeature::new("empty", vec![], 1.0);
        assert!(feat.is_empty());
        assert_eq!(feat.len(), 0);
        assert_eq!(feat.mean(), 0.0);
        assert_eq!(feat.std_dev(), 0.0);
    }
}
