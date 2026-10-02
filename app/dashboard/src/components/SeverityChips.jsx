import React from 'react';

const SEVERITY_COLORS = [
  'var(--sev-0)', // 0: Green
  'var(--sev-1)', // 1: Yellow-Green
  'var(--sev-2)', // 2: Yellow
  'var(--sev-3)', // 3: Orange
  'var(--sev-4)'  // 4: Red
];

const SEVERITY_LABELS = [
  'Ideal',
  'Barely Noticeable',
  'Clearly Noticeable',
  'Distracting',
  'Severely Botched'
];

export const getSeverityColor = (score) => {
  const index = Math.max(0, Math.min(4, Math.round(score)));
  return SEVERITY_COLORS[index];
};

export const getSeverityLabel = (score) => {
  const index = Math.max(0, Math.min(4, Math.round(score)));
  return SEVERITY_LABELS[index];
};

export const SeverityChip = ({ score }) => {
  const color = getSeverityColor(score);
  const label = getSeverityLabel(score);

  return (
    <span 
      className="severity-chip" 
      style={{ backgroundColor: color }}
      title={`Score: ${score}`}
    >
      {label}
    </span>
  );
};
