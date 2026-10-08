import React, { useState } from 'react';
import Header from './components/Header';
import AudioUploader from './components/AudioUploader';
import DualWaveform from './components/DualWaveform';
import FeatureTimelines from './components/FeatureTimelines';
import ExplanationPanel from './components/ExplanationPanel';
import RubricScores from './components/RubricScores';
import LimitationsPanel from './components/LimitationsPanel';
import { getDemoData, analyzeAudio, uploadAudio } from './api/client';
import { AlertCircle, Loader2 } from 'lucide-react';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [activePlayRegion, setActivePlayRegion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (idealFile, participantFile, transcript) => {
    setLoading(true);
    setError(null);
    try {
      const uploadId = await uploadAudio(idealFile, participantFile);
      const rawResult = await analyzeAudio(uploadId, transcript);
      setAnalysisData(rawResult);
    } catch (err) {
      console.error(err);
      setError('Unable to reach backend processing pipeline. You can explore calibrated benchmark data using the demo mode below.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getDemoData();
      setAnalysisData(data);
    } catch (err) {
      setError('Failed to load demo sample.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegionClick = (start, end) => {
    setActivePlayRegion({ start, end, nonce: Date.now() });
  };

  const handleReset = () => {
    setAnalysisData(null);
    setActivePlayRegion(null);
    setError(null);
  };

  return (
    <div className="dashboard-container">
      {/* 1. Header with SpeechMirror branding & Start Over */}
      <Header
        analysisData={analysisData}
        onReset={handleReset}
      />

      {/* Uploader View */}
      {!analysisData && !loading && (
        <AudioUploader
          onUpload={handleUpload}
          onDemo={handleDemo}
          isProcessing={loading}
        />
      )}

      {/* Loading State */}
      {loading && (
        <div className="panel" style={{ textAlign: 'center', padding: '60px 20px', maxWidth: '600px', margin: '40px auto' }}>
          <Loader2 size={36} style={{ animation: 'spin 1.5s linear infinite', color: '#D4902B', marginBottom: '14px' }} />
          <h2 style={{ fontFamily: 'Newsreader, serif', margin: '0 0 8px 0', fontSize: '1.4rem' }}>
            Analyzing Speech Alignment...
          </h2>
          <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Extracting frame acoustic features, mapping word intervals, and computing contrastive deltas.
          </p>
        </div>
      )}

      {/* Error Notice */}
      {error && (
        <div
          className="panel"
          style={{
            maxWidth: '680px',
            margin: '24px auto',
            backgroundColor: '#FFF5F5',
            borderColor: 'var(--accent-coral)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-coral-deep)', marginBottom: '8px' }}>
            <AlertCircle size={20} />
            <h3 style={{ margin: 0, fontSize: '1rem' }}>Connection Notice</h3>
          </div>
          <p style={{ margin: '0 0 14px 0', fontSize: '0.88rem', color: 'var(--text-primary)' }}>{error}</p>
          <button className="btn-playback" onClick={handleDemo}>
            Load Calibrated Demo Take
          </button>
        </div>
      )}

      {/* Analysis Page Sections */}
      {analysisData && (
        <>
          {/* 1. Performance Hero */}
          <RubricScores scores={analysisData.scores || analysisData.overall_scores} />

          {/* 2. Contrastive Speech Waveform Section & Word Alignment Strip */}
          <DualWaveform
            idealUrl={analysisData.idealAudioUrl}
            participantUrl={analysisData.participantAudioUrl}
            regions={analysisData.regions}
            activePlayRegion={activePlayRegion}
            onRegionSelected={(r) => handleRegionClick(r.start, r.end)}
          />

          {/* 3. Two-Column Analytics Layout: Acoustic Signals + Findings */}
          <div className="analytics-grid-two-col">
            {/* Left: Continuous Acoustic Signals */}
            <FeatureTimelines
              timelines={analysisData.timelines}
              regions={analysisData.regions}
            />

            {/* Right: Compact Findings Panel */}
            <ExplanationPanel
              regions={analysisData.regions}
              onRegionClick={handleRegionClick}
            />
          </div>

          {/* 4. Honest Limitations Panel */}
          <LimitationsPanel alignment={analysisData.alignment} />
        </>
      )}
    </div>
  );
}

export default App;
