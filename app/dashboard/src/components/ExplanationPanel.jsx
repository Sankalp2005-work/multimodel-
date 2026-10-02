import React from 'react';
import { SeverityChip } from './SeverityChips';
import { Activity, Clock, MicOff, VolumeX } from 'lucide-react';

const FLAW_ICONS = {
  rushed_pace: <Activity size={16} color="var(--flaw-rushed)" />,
  dead_pause: <Clock size={16} color="var(--flaw-pause)" />,
  flat_pitch: <MicOff size={16} color="var(--flaw-pitch)" />,
  mumbled_clarity: <VolumeX size={16} color="var(--flaw-mumbled)" />
};

const FLAW_LABELS = {
  rushed_pace: 'Rushed Pace',
  dead_pause: 'Dead Pause',
  flat_pitch: 'Flat Pitch',
  mumbled_clarity: 'Mumbled Clarity'
};

const ExplanationPanel = ({ regions, onRegionClick }) => {
  if (!regions || regions.length === 0) {
    return (
      <div className="panel">
        <h2>Detected Flaws</h2>
        <p style={{ color: 'var(--text-muted)' }}>No notable deviations detected.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Detected Flaws ({regions.length})</h2>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {regions.map((region) => (
          <div 
            key={region.id} 
            className="explanation-card"
            onClick={() => onRegionClick && onRegionClick(region.start, region.end)}
          >
            <div className="explanation-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}>
                {FLAW_ICONS[region.type] || <Activity size={16} />}
                {FLAW_LABELS[region.type] || region.type}
              </div>
              <SeverityChip score={region.severity} />
            </div>
            
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '8px' }}>
              {region.start.toFixed(2)}s – {region.end.toFixed(2)}s (Conf: {(region.confidence * 100).toFixed(0)}%)
            </div>
            
            <p style={{ margin: '8px 0', fontSize: '0.95rem' }}>
              {region.explanation}
            </p>
            
            {region.evidence && region.evidence.length > 0 && (
              <table className="explanation-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Ref</th>
                    <th>Observed</th>
                    <th>Δ %</th>
                    <th>Z-Score</th>
                  </tr>
                </thead>
                <tbody>
                  {region.evidence.map((ev, i) => (
                    <tr key={i}>
                      <td>{ev.feature}</td>
                      <td>{ev.ref.toFixed(2)}</td>
                      <td>{ev.obs.toFixed(2)}</td>
                      <td style={{ color: ev.delta > 0 ? 'var(--flaw-rushed)' : 'var(--flaw-pitch)' }}>
                        {ev.delta > 0 ? '+' : ''}{ev.delta.toFixed(1)}%
                      </td>
                      <td>{ev.z_score.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExplanationPanel;
