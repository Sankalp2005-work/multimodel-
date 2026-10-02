import React, { useState } from 'react';
import { ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react';

const LimitationsPanel = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="panel limitations-panel" style={{ borderLeft: '4px solid var(--text-muted)' }}>
      <div 
        className="explanation-header" 
        style={{ cursor: 'pointer', marginBottom: isOpen ? '16px' : '0' }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.1rem' }}>
          <AlertTriangle size={18} color="var(--text-muted)" />
          System Limitations (FR-10)
        </h2>
        {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </div>
      
      {isOpen && (
        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
          <p style={{ marginTop: 0 }}>Please interpret the analytics with the following constraints in mind:</p>
          <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
            <li><strong>Mic Sensitivity Variations:</strong> Differences in recording hardware may slightly skew energy and clarity metrics, despite algorithmic normalization.</li>
            <li><strong>Alignment Error Propagation:</strong> The system relies on dynamic time warping. Severe mispronunciations might misalign the timeline, causing false causal explanations.</li>
            <li><strong>No Semantic/Emotion Judgment:</strong> Analysis is purely acoustic. We do not evaluate the emotional appropriateness or semantic meaning of the delivery.</li>
            <li><strong>Speaker Normalization Assumptions:</strong> F0 (pitch) ranges are normalized using z-scores, which assumes the reference and participant have comparable expressive baselines.</li>
            <li><strong>Synthetic vs Real:</strong> The system highlights deviations from the reference. If the reference is synthetic (TTS), it may penalize natural human variations that are objectively acceptable.</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default LimitationsPanel;
