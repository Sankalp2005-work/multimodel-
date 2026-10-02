import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const client = axios.create({
  baseURL: API_BASE,
});

export const uploadAudio = async (idealFile, participantFile) => {
  const formData = new FormData();
  formData.append('ideal', idealFile);
  formData.append('participant', participantFile);
  
  const response = await client.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data.upload_id;
};

export const analyzeAudio = async (uploadId, transcript) => {
  const response = await client.post(`/analyze/${uploadId}`, { transcript });
  return response.data;
};

export const getResult = async (resultId) => {
  const response = await client.get(`/results/${resultId}`);
  return response.data;
};

// Mock data fallback for demo mode
export const getDemoData = async () => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        id: "demo-result-001",
        overall_scores: {
          pace: 3,
          pause: 1,
          pitch: 2,
          energy: 0,
          clarity: 4,
          overall: 2.0
        },
        regions: [
          {
            id: "r1",
            start: 1.5,
            end: 3.2,
            type: "rushed_pace",
            severity: 3,
            confidence: 0.92,
            explanation: "Participant spoke 45% faster than reference.",
            evidence: [
              { feature: "syllable_rate", ref: 4.1, obs: 5.95, delta: 45, z_score: 2.3 }
            ]
          },
          {
            id: "r2",
            start: 5.0,
            end: 7.1,
            type: "dead_pause",
            severity: 2,
            confidence: 0.88,
            explanation: "Extended unnatural silence detected.",
            evidence: [
              { feature: "pause_duration", ref: 0.5, obs: 2.1, delta: 320, z_score: 1.8 }
            ]
          },
          {
            id: "r3",
            start: 8.5,
            end: 11.0,
            type: "flat_pitch",
            severity: 4,
            confidence: 0.95,
            explanation: "Lack of intonation compared to expressive reference.",
            evidence: [
              { feature: "f0_std", ref: 35.2, obs: 8.4, delta: -76, z_score: -3.1 }
            ]
          },
          {
            id: "r4",
            start: 12.0,
            end: 14.5,
            type: "mumbled_clarity",
            severity: 3,
            confidence: 0.81,
            explanation: "Articulation degraded, vowels compressed.",
            evidence: [
              { feature: "mfcc_dist", ref: 0, obs: 4.2, delta: 100, z_score: 2.5 }
            ]
          }
        ],
        timelines: {
          time: Array.from({length: 100}, (_, i) => i * 0.15),
          ideal: {
            rate: Array.from({length: 100}, () => 3.5 + Math.random()),
            pitch: Array.from({length: 100}, () => 120 + Math.sin(Math.random()) * 20),
            energy: Array.from({length: 100}, () => 0.5 + Math.random() * 0.2)
          },
          participant: {
            rate: Array.from({length: 100}, (_, i) => (i > 10 && i < 21) ? 6.0 : 3.5 + Math.random()),
            pitch: Array.from({length: 100}, (_, i) => (i > 56 && i < 73) ? 100 : 120 + Math.sin(Math.random()) * 10),
            energy: Array.from({length: 100}, () => 0.4 + Math.random() * 0.3)
          }
        }
      });
    }, 800);
  });
};
