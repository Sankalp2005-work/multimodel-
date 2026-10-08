import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Play, FileAudio, CheckCircle2, X, Sparkles, ArrowRight } from 'lucide-react';

const SAMPLE_TRANSCRIPT = "The acoustic analyzer measures speech velocity across aligned phoneme intervals. Unexpected pauses disrupt listener comprehension, while flat cadence suppresses articulation resonance.";

const AudioUploader = ({ onUpload, onDemo, isProcessing = false }) => {
  const [idealFile, setIdealFile] = useState(null);
  const [participantFile, setParticipantFile] = useState(null);
  const [transcript, setTranscript] = useState('');

  const {
    getRootProps: getIdealProps,
    getInputProps: getIdealInputProps,
    isDragActive: isIdealDrag,
  } = useDropzone({
    accept: { 'audio/*': ['.wav', '.mp3', '.m4a'] },
    maxFiles: 1,
    onDrop: (accepted) => {
      if (accepted && accepted[0]) setIdealFile(accepted[0]);
    },
  });

  const {
    getRootProps: getParticipantProps,
    getInputProps: getParticipantInputProps,
    isDragActive: isPartDrag,
  } = useDropzone({
    accept: { 'audio/*': ['.wav', '.mp3', '.m4a'] },
    maxFiles: 1,
    onDrop: (accepted) => {
      if (accepted && accepted[0]) setParticipantFile(accepted[0]);
    },
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!idealFile || !participantFile || isProcessing) return;
    onUpload(idealFile, participantFile, transcript);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  };

  return (
    <div className="uploader-hero-box">
      {/* Title Header */}
      <div className="uploader-title-block">
        <span className="uploader-eyebrow">Acoustic Benchmark Pipeline</span>
        <h2 className="uploader-heading">Contrastive Speech Evaluation</h2>
        <p className="uploader-subtext">
          Upload an ideal baseline reference and a participant test recording to compute grounded phonetic deltas and prosodic deviations.
        </p>
      </div>

      {/* Dual Dropzones */}
      <div className="dropzones-pair-row">
        {/* Dropzone 1: Ideal Reference */}
        <div
          {...getIdealProps()}
          className={`dropzone-cell ${idealFile ? 'has-file' : ''} ${isIdealDrag ? 'is-drag-active' : ''}`}
        >
          <input {...getIdealInputProps()} />
          
          {idealFile ? (
            <div className="dropzone-file-info">
              <div className="dropzone-file-icon-wrap is-reference">
                <CheckCircle2 size={24} color="#5C7C58" />
              </div>
              <div className="dropzone-file-meta">
                <div className="dropzone-filename" title={idealFile.name}>{idealFile.name}</div>
                <div className="dropzone-filesize">{formatFileSize(idealFile.size)} · Ideal Reference</div>
              </div>
              <button
                type="button"
                className="dropzone-btn-remove"
                onClick={(e) => {
                  e.stopPropagation();
                  setIdealFile(null);
                }}
                title="Remove file"
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <>
              <div className="dropzone-upload-icon-wrap is-reference">
                <FileAudio size={28} />
              </div>
              <div className="dropzone-cell-label">Ideal Reference Audio</div>
              <div className="dropzone-cell-hint">
                Drag & drop or <span className="dropzone-browse-link">browse</span>
              </div>
              <div className="dropzone-format-pill">WAV, MP3, M4A · 16kHz Baseline</div>
            </>
          )}
        </div>

        {/* Dropzone 2: Participant Candidate */}
        <div
          {...getParticipantProps()}
          className={`dropzone-cell ${participantFile ? 'has-file' : ''} ${isPartDrag ? 'is-drag-active' : ''}`}
        >
          <input {...getParticipantInputProps()} />

          {participantFile ? (
            <div className="dropzone-file-info">
              <div className="dropzone-file-icon-wrap is-participant">
                <CheckCircle2 size={24} color="#C96F4A" />
              </div>
              <div className="dropzone-file-meta">
                <div className="dropzone-filename" title={participantFile.name}>{participantFile.name}</div>
                <div className="dropzone-filesize">{formatFileSize(participantFile.size)} · Candidate Take</div>
              </div>
              <button
                type="button"
                className="dropzone-btn-remove"
                onClick={(e) => {
                  e.stopPropagation();
                  setParticipantFile(null);
                }}
                title="Remove file"
              >
                <X size={14} />
              </button>
            </div>
          ) : (
            <>
              <div className="dropzone-upload-icon-wrap is-participant">
                <UploadCloud size={28} />
              </div>
              <div className="dropzone-cell-label">Participant Candidate Audio</div>
              <div className="dropzone-cell-hint">
                Drag & drop or <span className="dropzone-browse-link">browse</span>
              </div>
              <div className="dropzone-format-pill">WAV, MP3, M4A · Evaluation Take</div>
            </>
          )}
        </div>
      </div>

      {/* Transcript Input */}
      <div className="uploader-transcript-field">
        <div className="transcript-label-row">
          <label className="transcript-label" htmlFor="transcript-input">
            Shared Transcript <span className="transcript-badge">Recommended for Forced Alignment</span>
          </label>
          <button
            type="button"
            className="transcript-sample-btn"
            onClick={() => setTranscript(SAMPLE_TRANSCRIPT)}
          >
            Insert Benchmark Script
          </button>
        </div>
        <textarea
          id="transcript-input"
          className="transcript-textarea"
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="Paste or type matching speech transcript to enable phoneme-level forced alignment..."
          rows={3}
        />
      </div>

      {/* Primary Submit Button */}
      <div className="uploader-actions-row">
        <button
          type="button"
          className="btn-run-analysis"
          onClick={handleSubmit}
          disabled={!idealFile || !participantFile || isProcessing}
        >
          <span>{isProcessing ? 'Analyzing Acoustic Alignments...' : 'Run SpeechMirror Analysis'}</span>
          <ArrowRight size={16} />
        </button>
      </div>

      {/* Preset / Demo Explore Strip */}
      <div className="demo-trigger-strip">
        <div className="demo-trigger-label">
          <Sparkles size={14} color="var(--accent-terracotta)" />
          <span>Or explore pre-calibrated evaluation dataset:</span>
        </div>
        <button
          type="button"
          className="btn-demo-pill"
          onClick={onDemo}
          disabled={isProcessing}
        >
          <Play size={12} fill="currentColor" />
          <span>Load Calibrated Benchmark Sample (4 Deviations)</span>
        </button>
      </div>
    </div>
  );
};

export default AudioUploader;
