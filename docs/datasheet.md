# SpeechMirror Datasheet

## 1. Purpose
The SpeechMirror dataset was created to train and evaluate the SpeechMirror contrastive speech analytics system for the IIT Mandi Multimodal AI Hackathon (Track C). It provides paired "ideal" and "flawed" audio recordings of identical transcripts to facilitate word-aligned feature comparison and precise temporal flaw detection.

## 2. Collection Process
Recordings were collected in a controlled, quiet environment using consistent microphone setups per speaker. 
- **Ideal takes**: Speakers read the transcripts naturally.
- **Real Flawed takes**: Speakers intentionally performed specific flaws (`rushed_pace`, `dead_pause`, `flat_pitch`, `mumbled_clarity`) on designated target sentences.
- **Synthetic Flawed takes**: Generated deterministically from the ideal takes using DSP techniques.

## 3. Annotation
Flaw regions were labeled with exact timestamps (start/end) and flaw types. Real recordings underwent dual-annotation and acoustic verification to ensure a Label Accuracy (A1) of ≥ 95%.

## 4. Consent
All speakers provided explicit, written consent for their voice to be recorded, evaluated, and potentially shared publicly under an open license (e.g., CC0/CC-BY) as part of this dataset.

## 5. Limitations
See `limitations.md` for a comprehensive list of system limitations, including microphone variations, alignment error propagation, and the acoustic-only nature of the analysis.

## 6. License
The audio data and annotations are released under the CC-BY 4.0 license, unless otherwise noted.
