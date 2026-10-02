import React, { useState } from 'react';
import AudioUploader from './components/AudioUploader';
import DualWaveform from './components/DualWaveform';
import FeatureTimelines from './components/FeatureTimelines';
import ExplanationPanel from './components/ExplanationPanel';
import RubricScores from './components/RubricScores';
import LimitationsPanel from './components/LimitationsPanel';
import { getDemoData, analyzeAudio, uploadAudio } from './api/client';
import { Mic2 } from 'lucide-react';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [jumpTime, setJumpTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (idealFile, participantFile, transcript) => {
    setLoading(true);
    setError(null);
    try {
      const uploadId = await uploadAudio(idealFile, participantFile);
      const data = await analyzeAudio(uploadId, transcript);
      setAnalysisData(data);
    } catch (err) {
      console.error(err);
      setError("Failed to process audio. Please ensure backend is running or try the demo mode.");
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
      setError("Failed to load demo data.");
    } finally {
      setLoading(false);
    }
  };

  const handleRegionClick = (start, end) => {
    setJumpTime(start);
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>
          <Mic2 size={32} color="var(--sev-0)" />
          SpeechMirror
        </h1>
        <div style={{ color: 'var(--text-muted)' }}>Contrastive Speech Analytics</div>
      </header>

      {!analysisData && !loading && (
        <AudioUploader onUpload={handleUpload} onDemo={handleDemo} />
      )}

      {loading && (
        <div className="panel" style={{ textAlign: 'center', padding: '60px 20px' }}>
          <h2>Analyzing Speech...</h2>
          <p style={{ color: 'var(--text-muted)' }}>Aligning audio, extracting features, and detecting flaws.</p>
        </div>
      )}

      {error && (
        <div className="panel" style={{ borderColor: 'var(--sev-4)', backgroundColor: 'rgba(218, 54, 51, 0.1)' }}>
          <h2 style={{ color: 'var(--sev-4)' }}>Error</h2>
          <p>{error}</p>
          <button className="demo-btn" onClick={handleDemo}>Load Demo Instead</button>
        </div>
      )}

      {analysisData && (
        <>
          <div style={{ marginBottom: '20px' }}>
            <button className="demo-btn" onClick={() => setAnalysisData(null)}>
              &larr; Start Over
            </button>
          </div>

          <DualWaveform 
            regions={analysisData.regions} 
            jumpTime={jumpTime}
            idealUrl={analysisData.idealAudioUrl}
            participantUrl={analysisData.participantAudioUrl}
          />
          
          <div className="main-grid">
            <div className="left-column">
              <FeatureTimelines timelines={analysisData.timelines} regions={analysisData.regions} />
              <LimitationsPanel />
            </div>
            
            <div className="right-column">
              <RubricScores scores={analysisData.overall_scores} />
              <ExplanationPanel regions={analysisData.regions} onRegionClick={handleRegionClick} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
