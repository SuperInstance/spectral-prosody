use crate::error::ProsodyError;
use crate::rhythm::RhythmLayer;
use crate::prosody::ProsodyNode;

/// A simple MIDI note representation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MidiNote {
    pub pitch: u8,       // MIDI note number 0-127
    pub velocity: u8,    // 0-127
    pub start_time: f64, // seconds
    pub duration: f64,   // seconds
    pub channel: u8,     // 0-15
}

/// Export rhythmic layers as MIDI note patterns.
pub fn layers_to_midi(
    layers: &[RhythmLayer],
    nodes: &[ProsodyNode],
    bpm: f64,
) -> Result<Vec<MidiNote>, ProsodyError> {
    if layers.is_empty() {
        return Err(ProsodyError::EmptyFeature);
    }
    if nodes.is_empty() {
        return Err(ProsodyError::InsufficientNodes { got: 0, need: 1 });
    }

    let beat_duration = 60.0 / bpm; // seconds per beat
    let mut notes = Vec::new();

    for (channel, layer) in layers.iter().enumerate() {
        let ch = (channel as u8).min(15);

        // Map eigenvector values to note events
        let vec_abs_max = layer
            .eigenvector
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);

        for (i, &val) in layer.eigenvector.iter().enumerate() {
            if i >= nodes.len() {
                break;
            }

            // Only emit notes where eigenvector amplitude is above threshold
            let normalized = val.abs() / vec_abs_max;
            if normalized < 0.3 {
                continue;
            }

            let node = &nodes[i];
            let pitch = frequency_to_midi(node.pitch).min(127.0) as u8;
            let velocity = (normalized * 100.0).min(127.0) as u8;
            let start_time = node.time;
            let duration = node.duration.max(beat_duration * 0.25);

            notes.push(MidiNote {
                pitch,
                velocity,
                start_time,
                duration,
                channel: ch,
            });
        }
    }

    // Sort by start time
    notes.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    Ok(notes)
}

/// Convert a frequency in Hz to a MIDI note number.
pub fn frequency_to_midi(freq: f64) -> f64 {
    if freq <= 0.0 {
        return 0.0;
    }
    12.0 * (freq / 440.0).log2() + 69.0
}

/// Convert a MIDI note number to a frequency in Hz.
pub fn midi_to_frequency(note: u8) -> f64 {
    440.0 * 2.0_f64.powf((note as f64 - 69.0) / 12.0)
}

/// Export MIDI notes to a simple text-based MIDI file representation.
pub fn notes_to_csv(notes: &[MidiNote]) -> String {
    let mut csv = String::from("pitch,velocity,start_time,duration,channel\n");
    for note in notes {
        csv.push_str(&format!(
            "{},{},{:.4},{:.4},{}\n",
            note.pitch, note.velocity, note.start_time, note.duration, note.channel
        ));
    }
    csv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prosody::ProsodyNode;

    fn make_nodes(n: usize) -> Vec<ProsodyNode> {
        (0..n)
            .map(|i| ProsodyNode::new(i as f64 * 0.5, 1.0, 220.0 + i as f64 * 10.0, 0.25, 3000.0))
            .collect()
    }

    fn make_layers(n: usize) -> Vec<RhythmLayer> {
        (0..n)
            .map(|i| RhythmLayer {
                eigenvalue: (i + 1) as f64,
                eigenvector: (0..10).map(|j| ((i + j) as f64 * 0.1).sin()).collect(),
                period: 2.0 / (i + 1) as f64,
                strength: 1.0 / (i + 1) as f64,
            })
            .collect()
    }

    #[test]
    fn test_frequency_to_midi() {
        // A440 = MIDI 69
        assert!((frequency_to_midi(440.0) - 69.0).abs() < 0.01);
        // Middle C ≈ 261.63 Hz = MIDI 60
        assert!((frequency_to_midi(261.63) - 60.0).abs() < 0.01);
        // Zero/negative → 0
        assert_eq!(frequency_to_midi(0.0), 0.0);
        assert_eq!(frequency_to_midi(-1.0), 0.0);
    }

    #[test]
    fn test_midi_to_frequency() {
        assert!((midi_to_frequency(69) - 440.0).abs() < 0.01);
        assert!((midi_to_frequency(60) - 261.63).abs() < 0.1);
    }

    #[test]
    fn test_midi_roundtrip() {
        for note in (21u8..108).step_by(7) {
            let freq = midi_to_frequency(note);
            let back = frequency_to_midi(freq);
            assert!((back - note as f64).abs() < 0.01, "note {note}: {back}");
        }
    }

    #[test]
    fn test_layers_to_midi_basic() {
        let nodes = make_nodes(10);
        let layers = make_layers(3);
        let notes = layers_to_midi(&layers, &nodes, 120.0).unwrap();
        assert!(!notes.is_empty());
        // Notes should be sorted by start time
        for w in notes.windows(2) {
            assert!(w[0].start_time <= w[1].start_time);
        }
    }

    #[test]
    fn test_layers_to_midi_pitch_range() {
        let nodes = make_nodes(5);
        let layers = make_layers(1);
        let notes = layers_to_midi(&layers, &nodes, 120.0).unwrap();
        for note in &notes {
            assert!(note.pitch <= 127);
            assert!(note.velocity <= 127);
            assert!(note.channel <= 15);
        }
    }

    #[test]
    fn test_layers_to_midi_empty_layers() {
        let nodes = make_nodes(5);
        assert!(layers_to_midi(&[], &nodes, 120.0).is_err());
    }

    #[test]
    fn test_layers_to_midi_empty_nodes() {
        let layers = make_layers(1);
        assert!(layers_to_midi(&layers, &[], 120.0).is_err());
    }

    #[test]
    fn test_notes_to_csv() {
        let notes = vec![
            MidiNote { pitch: 60, velocity: 100, start_time: 0.0, duration: 0.5, channel: 0 },
            MidiNote { pitch: 64, velocity: 80, start_time: 0.5, duration: 0.5, channel: 0 },
        ];
        let csv = notes_to_csv(&notes);
        assert!(csv.contains("pitch,velocity"));
        assert!(csv.contains("60,100"));
        assert!(csv.contains("64,80"));
    }

    #[test]
    fn test_channel_limiting() {
        // More layers than 16 channels → channel capped at 15
        let nodes = make_nodes(5);
        let layers: Vec<RhythmLayer> = (0..20)
            .map(|i| RhythmLayer {
                eigenvalue: i as f64,
                eigenvector: vec![1.0; 5],
                period: 1.0,
                strength: 0.5,
            })
            .collect();
        let notes = layers_to_midi(&layers, &nodes, 120.0).unwrap();
        for note in &notes {
            assert!(note.channel <= 15);
        }
    }
}
