import React, { useState } from 'react';
import { FLAW_METADATA } from './SeverityChips';
import { Play, CheckCircle2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';

const ExplanationPanel = ({ regions = [], onRegionClick = null }) => {
  const [filterType, setFilterType] = useState('all');
  const [expandedId, setExpandedId] = useState(null);

  const filteredRegions = (regions || []).filter((r) => {
    if (filterType === 'all') return true;
    return r.type === filterType;
  });

  const countStr = String(regions.length).padStart(2, '0');

  const formatDelta = (delta) => {
    if (delta === undefined || delta === null || isNaN(delta)) return '—';
    const num = Number(delta);
    return `${num > 0 ? '+' : ''}${num.toFixed(1)}%`;
  };

  const formatZ = (z) => {
    if (z === undefined || z === null || isNaN(z)) return '—';
    const num = Number(z);
    return `${num > 0 ? '+' : ''}${num.toFixed(2)}σ`;
  };

  const formatVal = (val) => {
    if (val === undefined || val === null || isNaN(val)) return '—';
    return Number(val).toFixed(2);
  };

  const getCleanLabel = (type) => {
    if (FLAW_METADATA[type]?.label) return FLAW_METADATA[type].label;
    if (!type) return 'Deviation';
    return type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const getIcon = (type) => {
    const meta = FLAW_METADATA[type];
    return meta ? meta.icon : AlertCircle;
  };

  return (
    <div className="findings-module">
      {/* Conceptual Header */}
      <div className="findings-header-block">
        <div>
          <span className="findings-eyebrow">Acoustic Deviations</span>
          <h3 className="findings-title">FINDINGS</h3>
        </div>
        <span className="deviations-badge">
          {countStr} deviations detected
        </span>
      </div>

      {/* Filter Tabs if multiple flaws */}
      {regions.length > 1 && (
        <div className="timelines-tab-row" style={{ marginBottom: '14px', flexWrap: 'wrap' }}>
          <button
            className={`timeline-tab-btn ${filterType === 'all' ? 'active' : ''}`}
            onClick={() => setFilterType('all')}
          >
            All ({regions.length})
          </button>
          {Object.entries(FLAW_METADATA).map(([key, meta]) => {
            const count = regions.filter((r) => r.type === key).length;
            if (count === 0) return null;
            return (
              <button
                key={key}
                className={`timeline-tab-btn ${filterType === key ? 'active' : ''}`}
                onClick={() => setFilterType(key)}
              >
                {meta.label} ({count})
              </button>
            );
          })}
        </div>
      )}

      {/* Clean Take: 0 Flaws */}
      {regions.length === 0 ? (
        <div
          style={{
            textAlign: 'center',
            padding: '32px 18px',
            backgroundColor: 'var(--bg-cream-soft)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <CheckCircle2 size={28} color="#5C7C58" style={{ marginBottom: '8px' }} />
          <h4 style={{ margin: '0 0 4px 0', color: 'var(--text-primary)', fontSize: '0.95rem' }}>
            No Deviations Detected
          </h4>
          <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Candidate recording conforms closely to reference baseline tolerances.
          </p>
        </div>
      ) : filteredRegions.length === 0 ? (
        <div style={{ padding: '18px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
          No findings for the selected filter category.
        </div>
      ) : (
        /* Compact Finding Cards List */
        <div className="findings-cards-list">
          {filteredRegions.map((region, idx) => {
            const IconComp = getIcon(region.type);
            const typeLabel = getCleanLabel(region.type);
            const scoreNum = Number(region.severity !== undefined ? region.severity : 3.0);
            const startTime = Number(region.start || 0).toFixed(2);
            const endTime = Number(region.end || 0).toFixed(2);
            const isExpanded = expandedId === (region.id || idx);

            return (
              <div
                key={region.id || idx}
                className="finding-card-compact"
                onClick={() => onRegionClick && onRegionClick(region.start, region.end)}
                style={{ cursor: 'pointer' }}
              >
                {/* Top Line: Icon + Type Name & Severity Score */}
                <div className="finding-top-line">
                  <div className="finding-type-title">
                    <IconComp size={15} color="var(--accent-terracotta)" />
                    <span>{typeLabel}</span>
                  </div>

                  <span className="finding-score-chip" title="Severity Rating">
                    {scoreNum.toFixed(1)}
                  </span>
                </div>

                {/* Timestamp Row */}
                <div className="finding-time-span">
                  {startTime} — {endTime}s
                </div>

                {/* Short Explanation */}
                <div className="finding-short-explanation">
                  {region.explanation || 'Measurable deviation from calibrated reference delivery.'}
                </div>

                {/* Action Buttons Row */}
                <div className="finding-actions-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  {region.evidence && region.evidence.length > 0 ? (
                    <button
                      type="button"
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-secondary)',
                        fontSize: '0.72rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '3px',
                        cursor: 'pointer',
                        padding: 0,
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedId(isExpanded ? null : (region.id || idx));
                      }}
                    >
                      {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      {isExpanded ? 'Hide metrics' : 'View metrics'}
                    </button>
                  ) : <div />}

                  <button
                    className="btn-play-region-compact"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (onRegionClick) onRegionClick(region.start, region.end);
                    }}
                    title={`Play region ${startTime}s - ${endTime}s`}
                  >
                    <Play size={11} fill="currentColor" /> Play Region
                  </button>
                </div>

                {/* Expanded Micro Evidence Table */}
                {isExpanded && region.evidence && region.evidence.length > 0 && (
                  <div
                    style={{
                      marginTop: '10px',
                      paddingTop: '8px',
                      borderTop: '1px solid var(--border-subtle)',
                      fontSize: '0.75rem',
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono, monospace)' }}>
                      <thead>
                        <tr style={{ color: 'var(--text-tertiary)', textAlign: 'left', fontSize: '0.68rem', textTransform: 'uppercase' }}>
                          <th style={{ paddingBottom: '4px' }}>Metric</th>
                          <th style={{ paddingBottom: '4px' }}>Ref</th>
                          <th style={{ paddingBottom: '4px' }}>Obs</th>
                          <th style={{ paddingBottom: '4px' }}>Delta</th>
                          <th style={{ paddingBottom: '4px' }}>Z-Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {region.evidence.map((ev, evIdx) => {
                          const deltaVal = ev.delta_pct !== undefined ? ev.delta_pct : ev.delta;
                          const zVal = ev.z !== undefined ? ev.z : ev.z_score;
                          return (
                            <tr key={evIdx} style={{ color: 'var(--text-primary)' }}>
                              <td style={{ padding: '2px 0', fontFamily: 'inherit', fontWeight: 500 }}>{ev.feature}</td>
                              <td style={{ padding: '2px 0', color: 'var(--text-secondary)' }}>{formatVal(ev.ref)}</td>
                              <td style={{ padding: '2px 0', color: 'var(--text-secondary)' }}>{formatVal(ev.obs)}</td>
                              <td style={{ padding: '2px 0', color: 'var(--accent-terracotta)', fontWeight: 600 }}>{formatDelta(deltaVal)}</td>
                              <td style={{ padding: '2px 0', color: 'var(--text-secondary)' }}>{formatZ(zVal)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ExplanationPanel;
