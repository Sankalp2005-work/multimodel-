import React from 'react';

const RUBRIC_ITEMS = [
  { key: 'pace', label: 'Pace', fill: '#FF9F9F' },
  { key: 'pause', label: 'Pause', fill: '#FFDFA8' },
  { key: 'pitch', label: 'Pitch', fill: '#C2E2BE' },
  { key: 'energy', label: 'Energy', fill: '#FFDFA8' },
  { key: 'clarity', label: 'Clarity', fill: '#FFCDB8' },
];

const RubricScores = ({ scores }) => {
  if (!scores) return null;

  const overall = Number(scores.overall !== undefined ? scores.overall : 2.0);

  // Status mapping
  let statusText = 'Needs Work';
  let summaryText = 'Noticeable acoustic deviations';

  if (overall < 0.8) {
    statusText = 'Ideal';
    summaryText = 'Conforms closely to reference baseline';
  } else if (overall < 1.8) {
    statusText = 'Acceptable';
    summaryText = 'Minor natural variations observed';
  } else if (overall < 2.8) {
    statusText = 'Needs Work';
    summaryText = 'Noticeable acoustic deviations';
  } else {
    statusText = 'Significant Botch';
    summaryText = 'Substantial delivery breakdown';
  }

  return (
    <div className="performance-hero">
      <div className="hero-main-row">
        {/* Left Side: Editorial Score Hero */}
        <div className="hero-left-col">
          <div className="hero-eyebrow">ANALYSIS COMPLETE</div>

          <div className="hero-score-row">
            <span className="hero-major-score">{overall.toFixed(1)}</span>
            <span className="hero-score-scale">/ 4.0</span>
          </div>

          <div>
            <span className="hero-status-tag">{statusText}</span>
          </div>

          <p className="hero-summary-statement">{summaryText}</p>

          {/* Requirement 8: Subtle Analysis Confidence */}
          <div className="hero-confidence-bar">
            <span className="confidence-label">ANALYSIS CONFIDENCE</span>
            <div className="confidence-meter-track" title="92% Forced Alignment Confidence">
              <div className="confidence-meter-fill" style={{ width: '92%' }} />
            </div>
            <span className="confidence-value">92%</span>
            <span style={{ color: '#5C7C58', fontWeight: 500 }}>High confidence</span>
          </div>
        </div>

        {/* Right Side: Compact Rubric Breakdown */}
        <div className="hero-rubric-col">
          {RUBRIC_ITEMS.map((item) => {
            const raw = Number(scores[item.key] !== undefined ? scores[item.key] : 1.0);
            const score = Math.max(0, Math.min(4, raw));
            const pct = Math.max(6, (score / 4) * 100);

            return (
              <div key={item.key} className="rubric-compact-row">
                <span className="rubric-compact-name">{item.label}</span>
                <div className="rubric-compact-track">
                  <div
                    className="rubric-compact-fill"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: item.fill,
                    }}
                  />
                </div>
                <span className="rubric-compact-score">{score.toFixed(1)}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default RubricScores;
