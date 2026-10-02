# SpeechMirror Honest Limitations

While SpeechMirror is highly effective at contrastive speech analysis, it has several limitations:

1. **Mic Sensitivity Variations**: If the participant uses a microphone with a significantly different frequency response or gain staging than the ideal reference, the feature normalization might not fully compensate, leading to potential false positives in the Energy and Clarity dimensions.
2. **Alignment Error Propagation**: The accuracy of the flaw region detection is fundamentally bounded by the accuracy of the underlying forced alignment model (WhisperX). In cases of severe mumbling or background noise, alignment may fail or become inaccurate, which propagates into timestamp errors for the detected regions.
3. **No Semantic or Emotional Judgment**: The system only analyzes acoustic features (pitch, pace, pauses, energy, clarity). It does not understand the meaning of the words, nor can it evaluate the appropriate emotional delivery of the speech.
4. **Speaker Normalization Assumptions**: While z-score and ratio-based normalizations are used, comparing speakers with drastically different natural baselines (e.g., very different speaking rates) against a single ideal reference can sometimes lead to miscalibrated severity estimates.
5. **Synthetic vs. Real Differences**: Synthetic perturbations provide exact ground truth for training and tuning, but human-performed flaws often contain subtle co-articulation changes that DSP techniques cannot perfectly mimic. Performance on synthetic data may overestimate real-world performance.
