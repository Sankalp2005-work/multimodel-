import React, { useState } from 'react';
import { AlertTriangle, ShieldAlert, ChevronDown, ChevronUp } from 'lucide-react';

const SUMMARY_TAGS = [
  'Acoustic analysis only',
  'Speaker variation',
  'Recording conditions',
  'Reference quality',
];

const DETAILED_CONSTRAINTS = [
  {
    id: 'acoustic',
    title: 'Acoustic analysis only',
    desc: 'Analysis evaluates signal-level acoustic deviations (prosody, timing, energy). It does not grade semantic meaning, emotional authenticity, or persuasive intent.',
  },
  {
    id: 'speaker',
    title: 'Speaker variation & normalization',
    desc: 'Pitch (F0) is normalized using statistical z-scores. Natural anatomical vocal tract differences are separated from intentional intonation shifts.',
  },
  {
    id: 'recording',
    title: 'Recording conditions & hardware',
    desc: 'Differences in transducer response, room reverberation, or gain staging may modestly influence spectral centroid and energy measures.',
  },
  {
    id: 'reference',
    title: 'Reference baseline dependency',
    desc: 'Deviations are relative to the selected reference track. If comparing against synthetic TTS, natural human micro-variations may register as deltas.',
  },
];

const LimitationsPanel = ({ alignment = null }) => {
  const [showDetails, setShowDetails] = useState(false);

  const meanConf = alignment?.mean_conf !== undefined ? alignment.mean_conf : 0.94;
  const wer = alignment?.wer !== undefined ? alignment.wer : 0.028;
  const method = alignment?.method || 'WhisperX Forced Alignment';
  const isLowConfidence = meanConf < 0.75 || wer > 0.15;

  return (
    <div className="limitations-section">
      {/* Header Bar */}
      <div className="limitations-header-bar">
        <div className="limitations-icon-badge">
          <AlertTriangle size={15} />
        </div>
        <div>
          <h3 className="limitations-header-title">Honest Limitations</h3>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            System boundaries · {method} · WER {(wer * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Concise Summary Pills (Requirement 9) */}
      <div className="limitations-summary-chips-row">
        {SUMMARY_TAGS.map((tag) => (
          <span key={tag} className="limitations-tag-chip">
            {tag}
          </span>
        ))}
      </div>

      {/* Low-confidence abstain notice if triggered */}
      {isLowConfidence && (
        <div
          style={{
            backgroundColor: '#FFEAE8',
            border: '1px solid var(--accent-coral)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 14px',
            marginBottom: '14px',
            fontSize: '0.8rem',
            color: 'var(--accent-coral-deep)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <ShieldAlert size={16} />
          <span>
            <strong>Abstain Notice:</strong> Alignment confidence is below 75%. Reported flaw boundaries may reflect phoneme alignment drift rather than speech faults.
          </span>
        </div>
      )}

      {/* Concise core explanation statement */}
      <p style={{ margin: '0 0 12px 0', fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        SpeechMirror measures relative acoustic contrast against a calibrated ground-truth reference. Scores represent objective phonological and prosodic divergence, not subjective communication competence.
      </p>

      {/* Expand/Collapse Detailed Methodology */}
      <div>
        <button
          type="button"
          onClick={() => setShowDetails(!showDetails)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--accent-terracotta)',
            fontSize: '0.78rem',
            fontWeight: 600,
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            cursor: 'pointer',
            padding: 0,
            marginBottom: showDetails ? '12px' : 0,
          }}
        >
          {showDetails ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {showDetails ? 'Hide detailed methodology & constraints' : 'View detailed methodology & constraints'}
        </button>

        {showDetails && (
          <div className="limitations-disclosure-grid">
            {DETAILED_CONSTRAINTS.map((item) => (
              <div key={item.id} className="limitation-item-box">
                <div className="limitation-box-title">{item.title}</div>
                <p className="limitation-box-text">{item.desc}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default LimitationsPanel;
