import React from 'react';
import { Activity, Clock, MicOff, VolumeX, CheckCircle, AlertCircle } from 'lucide-react';

export const SEVERITY_COLORS = [
  '#2F6327', // 0: Ideal (Soft sage green)
  '#7D5618', // 1: Barely Noticeable (Warm apricot amber)
  '#9E4D1B', // 2: Clearly Noticeable (Warm peach copper)
  '#9C332A', // 3: Distracting (Warm coral terracotta)
  '#8C1C26', // 4: Severely Botched (Muted deep crimson)
];

export const SEVERITY_BG_COLORS = [
  '#EBF5EA', // 0
  '#FFF6DD', // 1
  '#FFF0E6', // 2
  '#FFEAE8', // 3
  '#FFDEE0', // 4
];

export const SEVERITY_BORDER_COLORS = [
  '#C2E2BE',
  '#FFE399',
  '#FFCDB8',
  '#FF9F9F',
  '#FFB0B8',
];

export const SEVERITY_LABELS = [
  'Ideal',
  'Barely Noticeable',
  'Clearly Noticeable',
  'Distracting',
  'Severely Botched',
];

export const FLAW_METADATA = {
  rushed_pace: {
    label: 'Rushed Pace',
    color: '#9C332A',
    bgColor: '#FFEAE8',
    borderColor: '#FFCDB8',
    icon: Activity,
    featureTarget: 'speech_rate_ratio',
    summary: 'Accelerated speaking velocity with compressed pauses.',
  },
  dead_pause: {
    label: 'Dead Pause',
    color: '#7D5618',
    bgColor: '#FFF6DD',
    borderColor: '#FFDFA8',
    icon: Clock,
    featureTarget: 'pause_duration_s',
    summary: 'Unnatural hesitation or silence mid-sentence.',
  },
  flat_pitch: {
    label: 'Flat Pitch',
    color: '#2F6327',
    bgColor: '#EBF5EA',
    borderColor: '#C2E2BE',
    icon: MicOff,
    featureTarget: 'f0_std_semitones',
    summary: 'Monotone intonation with flattened F0 variance.',
  },
  mumbled_clarity: {
    label: 'Mumbled Clarity',
    color: '#385A82',
    bgColor: '#EDF4FC',
    borderColor: '#C8DEF5',
    icon: VolumeX,
    featureTarget: 'spectral_centroid_hz',
    summary: 'Softened consonant articulation and spectral muffling.',
  },
};

export const getSeverityColor = (score) => {
  const num = Number(score);
  if (isNaN(num)) return SEVERITY_COLORS[0];
  const index = Math.max(0, Math.min(4, Math.round(num)));
  return SEVERITY_COLORS[index];
};

export const getSeverityLabel = (score) => {
  const num = Number(score);
  if (isNaN(num)) return SEVERITY_LABELS[0];
  const index = Math.max(0, Math.min(4, Math.round(num)));
  return SEVERITY_LABELS[index];
};

export const SeverityChip = ({ score, showValue = true }) => {
  const num = Number(score) || 0;
  const idx = Math.max(0, Math.min(4, Math.round(num)));
  const color = SEVERITY_COLORS[idx];
  const bg = SEVERITY_BG_COLORS[idx];
  const border = SEVERITY_BORDER_COLORS[idx];
  const label = SEVERITY_LABELS[idx];

  return (
    <span
      className={`severity-chip sev-chip-${idx}`}
      style={{
        backgroundColor: bg,
        borderColor: border,
        color: color,
      }}
      title={`Severity Score: ${num.toFixed(1)} / 4.0`}
    >
      <span>{label}</span>
      {showValue && <span style={{ opacity: 0.8, fontFamily: 'monospace' }}>[{num.toFixed(1)}]</span>}
    </span>
  );
};

export const FlawTypeBadge = ({ type, size = 15 }) => {
  const meta = FLAW_METADATA[type] || {
    label: type ? type.replace(/_/g, ' ') : 'Deviation',
    color: '#252525',
    bgColor: '#FFFDF7',
    borderColor: '#E8DDD2',
    icon: AlertCircle,
  };
  const IconComponent = meta.icon;

  return (
    <span
      className="flaw-type-tag"
      style={{
        backgroundColor: meta.bgColor,
        borderColor: meta.borderColor,
        color: meta.color,
      }}
    >
      <IconComponent size={size} />
      <span>{meta.label}</span>
    </span>
  );
};

export default SeverityChip;
