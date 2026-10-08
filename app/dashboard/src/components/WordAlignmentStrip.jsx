import React, { useState } from 'react';

const TRANSCRIPT_TOKENS = [
  { id: 'w1', word: 'The', start: 0.15, end: 0.40, isFlawed: false, flawType: null },
  { id: 'w2', word: 'acoustic', start: 0.42, end: 0.95, isFlawed: false, flawType: null },
  { id: 'w3', word: 'analyzer', start: 0.98, end: 1.48, isFlawed: false, flawType: null },
  { id: 'w4', word: 'measures', start: 1.50, end: 1.95, isFlawed: true, flawType: 'rushed_pace' },
  { id: 'w5', word: 'speech', start: 1.96, end: 2.38, isFlawed: true, flawType: 'rushed_pace' },
  { id: 'w6', word: 'velocity', start: 2.40, end: 2.88, isFlawed: true, flawType: 'rushed_pace' },
  { id: 'w7', word: 'across', start: 2.90, end: 3.38, isFlawed: true, flawType: 'rushed_pace' },
  { id: 'w8', word: 'aligned', start: 3.42, end: 3.98, isFlawed: false, flawType: null },
  { id: 'w9', word: 'phoneme', start: 4.02, end: 4.65, isFlawed: false, flawType: null },
  { id: 'w10', word: 'intervals.', start: 4.68, end: 5.18, isFlawed: false, flawType: null },
  { id: 'w11', word: 'pause', start: 5.20, end: 7.50, isFlawed: true, flawType: 'dead_pause', isPause: true },
  { id: 'w12', word: 'Unexpected', start: 7.55, end: 8.25, isFlawed: false, flawType: null },
  { id: 'w13', word: 'pauses', start: 8.28, end: 8.92, isFlawed: false, flawType: null },
  { id: 'w14', word: 'disrupt', start: 9.00, end: 9.65, isFlawed: true, flawType: 'flat_pitch' },
  { id: 'w15', word: 'listener', start: 9.68, end: 10.42, isFlawed: true, flawType: 'flat_pitch' },
  { id: 'w16', word: 'comprehension,', start: 10.45, end: 12.18, isFlawed: true, flawType: 'flat_pitch' },
  { id: 'w17', word: 'while', start: 12.22, end: 12.65, isFlawed: false, flawType: null },
  { id: 'w18', word: 'flat', start: 12.68, end: 13.05, isFlawed: false, flawType: null },
  { id: 'w19', word: 'cadence', start: 13.08, end: 13.48, isFlawed: false, flawType: null },
  { id: 'w20', word: 'suppresses', start: 13.50, end: 14.30, isFlawed: true, flawType: 'mumbled_clarity' },
  { id: 'w21', word: 'articulation', start: 14.32, end: 15.35, isFlawed: true, flawType: 'mumbled_clarity' },
  { id: 'w22', word: 'resonance.', start: 15.38, end: 16.78, isFlawed: true, flawType: 'mumbled_clarity' },
];

const WordAlignmentStrip = ({ onWordClick = null, activeTime = null }) => {
  const [selectedId, setSelectedId] = useState(null);
  const [hoveredToken, setHoveredToken] = useState(null);

  const handleClick = (token) => {
    setSelectedId(token.id);
    if (onWordClick) {
      onWordClick(token.start, token.end);
    }
  };

  return (
    <div className="word-alignment-strip">
      <div className="alignment-strip-header">
        <span className="alignment-strip-title">Word-Level Forced Alignment</span>
        <div className="alignment-strip-legend">
          <div className="legend-chip">
            <span className="dot-neutral" />
            <span>Neutral</span>
          </div>
          <div className="legend-chip">
            <span className="dot-flaw" />
            <span>Flawed (Coral)</span>
          </div>
          <div className="legend-chip">
            <span className="dot-selected" />
            <span>Selected (Terracotta)</span>
          </div>
        </div>
      </div>

      <div className="word-tokens-container">
        {TRANSCRIPT_TOKENS.map((token) => {
          const isSelected = selectedId === token.id;
          const isCurrentlyActive =
            activeTime !== null && activeTime >= token.start && activeTime <= token.end;

          let classes = 'word-token-item';
          if (token.isFlawed) classes += ' is-flawed';
          if (token.isPause) classes += ' is-pause';
          if (isSelected || isCurrentlyActive) classes += ' is-selected';

          return (
            <span
              key={token.id}
              className={classes}
              onClick={() => handleClick(token)}
              onMouseEnter={() => setHoveredToken(token)}
              onMouseLeave={() => setHoveredToken(null)}
              title={`${token.word} (${token.start.toFixed(2)}s – ${token.end.toFixed(2)}s)`}
            >
              {token.isFlawed && <span className="flaw-dot-indicator" />}
              <span>{token.isPause ? `[${token.word}]` : token.word}</span>
            </span>
          );
        })}
      </div>

      {hoveredToken && (
        <div style={{ marginTop: '8px', fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', gap: '12px' }}>
          <span>Word: <strong>{hoveredToken.word}</strong></span>
          <span>Time: <strong>{hoveredToken.start.toFixed(2)}s – {hoveredToken.end.toFixed(2)}s</strong></span>
          {hoveredToken.flawType && (
            <span style={{ color: 'var(--accent-terracotta)', fontWeight: 600 }}>
              Flaw: {hoveredToken.flawType.replace('_', ' ')}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default WordAlignmentStrip;
