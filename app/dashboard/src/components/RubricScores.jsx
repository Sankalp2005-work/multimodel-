import React from 'react';
import { getSeverityColor } from './SeverityChips';

const RubricScores = ({ scores }) => {
  if (!scores) return null;

  const renderScoreRow = (label, score, key) => {
    // Score is 0-4. 0 is best (green), 4 is worst (red).
    // Let's visualize this as a bar where lower is better, or just a filled segment.
    const percentage = (score / 4) * 100;
    const color = getSeverityColor(score);

    return (
      <div className="score-row" key={key}>
        <div className="score-label">{label}</div>
        <div className="score-bar-bg">
          <div 
            className="score-bar-fill" 
            style={{ 
              width: `${Math.max(5, percentage)}%`, 
              backgroundColor: color,
              transition: 'width 0.5s ease-out'
            }} 
          />
        </div>
        <div style={{ width: '30px', textAlign: 'right', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
          {score.toFixed(1)}
        </div>
      </div>
    );
  };

  const dimensions = [
    { key: 'pace', label: 'Pace' },
    { key: 'pause', label: 'Pause' },
    { key: 'pitch', label: 'Pitch' },
    { key: 'energy', label: 'Energy' },
    { key: 'clarity', label: 'Clarity' }
  ];

  return (
    <div className="panel">
      <h2>Overall Performance</h2>
      
      <div style={{ marginBottom: '24px', textAlign: 'center' }}>
        <div style={{ fontSize: '3rem', fontWeight: 'bold', color: getSeverityColor(scores.overall) }}>
          {scores.overall.toFixed(1)}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Aggregate Score (0=Ideal, 4=Botched)</div>
      </div>

      <div>
        {dimensions.map(dim => renderScoreRow(dim.label, scores[dim.key], dim.key))}
      </div>
    </div>
  );
};

export default RubricScores;
