import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, Play } from 'lucide-react';

const AudioUploader = ({ onUpload, onDemo }) => {
  const [idealFile, setIdealFile] = useState(null);
  const [participantFile, setParticipantFile] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const { getRootProps: getIdealProps, getInputProps: getIdealInputProps } = useDropzone({
    accept: {'audio/*': ['.wav', '.mp3', '.m4a']},
    maxFiles: 1,
    onDrop: accepted => setIdealFile(accepted[0])
  });

  const { getRootProps: getParticipantProps, getInputProps: getParticipantInputProps } = useDropzone({
    accept: {'audio/*': ['.wav', '.mp3', '.m4a']},
    maxFiles: 1,
    onDrop: accepted => setParticipantFile(accepted[0])
  });

  const handleSubmit = async () => {
    if (!idealFile || !participantFile) return;
    setIsUploading(true);
    await onUpload(idealFile, participantFile, transcript);
    setIsUploading(false);
  };

  return (
    <div className="panel">
      <h2>Upload Recordings</h2>
      <div className="dropzone-container">
        <div {...getIdealProps()} className="dropzone" style={{ borderColor: idealFile ? 'var(--sev-0)' : 'var(--border-color)' }}>
          <input {...getIdealInputProps()} />
          <UploadCloud size={32} color="var(--text-muted)" style={{ marginBottom: '10px' }} />
          <div>{idealFile ? idealFile.name : "Drop Ideal Reference Audio"}</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '5px' }}>WAV, MP3, M4A</div>
        </div>

        <div {...getParticipantProps()} className="dropzone" style={{ borderColor: participantFile ? 'var(--sev-0)' : 'var(--border-color)' }}>
          <input {...getParticipantInputProps()} />
          <UploadCloud size={32} color="var(--text-muted)" style={{ marginBottom: '10px' }} />
          <div>{participantFile ? participantFile.name : "Drop Participant Audio"}</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '5px' }}>WAV, MP3, M4A</div>
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-muted)' }}>Shared Transcript (Optional)</label>
        <textarea 
          value={transcript}
          onChange={(e) => setTranscript(e.target.value)}
          placeholder="Paste the transcript text here to improve alignment..."
          style={{ 
            width: '100%', height: '80px', padding: '10px', 
            background: 'var(--bg-card)', color: 'var(--text-main)', 
            border: '1px solid var(--border-color)', borderRadius: '6px'
          }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center' }}>
        <button 
          className="upload-btn" 
          onClick={handleSubmit} 
          disabled={!idealFile || !participantFile || isUploading}
          style={{ opacity: (!idealFile || !participantFile || isUploading) ? 0.5 : 1 }}
        >
          {isUploading ? 'Analyzing...' : 'Analyze Speech'}
        </button>
        <span style={{ margin: '0 15px', color: 'var(--text-muted)' }}>OR</span>
        <button className="demo-btn" onClick={onDemo} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Play size={16} /> Try Demo Data
        </button>
      </div>
    </div>
  );
};

export default AudioUploader;
