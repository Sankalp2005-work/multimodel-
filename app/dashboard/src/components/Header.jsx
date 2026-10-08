import React from 'react';
import { Mic2, RotateCcw } from 'lucide-react';

const Header = ({ analysisData, onReset }) => {
  return (
    <header className="editorial-header">
      {/* Brand & Subtitle */}
      <div className="brand-section">
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '32px',
            height: '32px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'var(--bg-peach-subtle)',
            color: 'var(--accent-terracotta)',
            border: '1px solid var(--accent-peach)',
            marginRight: '6px',
          }}
        >
          <Mic2 size={18} />
        </div>
        <h1 className="brand-title">SpeechMirror</h1>
        <span className="brand-subtitle">Contrastive Speech Analytics</span>
      </div>

      {/* Header Actions: Start Over Button */}
      {analysisData && (
        <div>
          <button className="header-action-btn" onClick={onReset} title="Reset analysis and load another take">
            <RotateCcw size={14} />
            <span>Start Over</span>
          </button>
        </div>
      )}
    </header>
  );
};

export default Header;
