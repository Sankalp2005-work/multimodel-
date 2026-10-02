import React, { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.esm.js';
import { Play, Pause, Square } from 'lucide-react';

const FLAW_COLORS = {
  rushed_pace: 'rgba(255, 107, 107, 0.4)',
  dead_pause: 'rgba(255, 217, 61, 0.4)',
  flat_pitch: 'rgba(107, 203, 119, 0.4)',
  mumbled_clarity: 'rgba(77, 150, 255, 0.4)'
};

const DualWaveform = ({ idealUrl, participantUrl, regions, jumpTime }) => {
  const idealContainerRef = useRef(null);
  const partContainerRef = useRef(null);
  const idealTimelineRef = useRef(null);
  const partTimelineRef = useRef(null);
  
  const idealWs = useRef(null);
  const partWs = useRef(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!idealContainerRef.current || !partContainerRef.current) return;

    // Initialize Ideal WaveSurfer
    idealWs.current = WaveSurfer.create({
      container: idealContainerRef.current,
      waveColor: '#4d96ff',
      progressColor: '#1e5fba',
      height: 100,
      normalize: true,
      plugins: [
        TimelinePlugin.create({ container: idealTimelineRef.current })
      ]
    });

    // Initialize Participant WaveSurfer
    partWs.current = WaveSurfer.create({
      container: partContainerRef.current,
      waveColor: '#ff6b6b',
      progressColor: '#ba2a2a',
      height: 100,
      normalize: true,
      plugins: [
        TimelinePlugin.create({ container: partTimelineRef.current })
      ]
    });

    const wsRegions = partWs.current.registerPlugin(RegionsPlugin.create());

    // Load audio (fallback to empty buffer if no URL for demo styling without actual audio files)
    if (idealUrl) idealWs.current.load(idealUrl);
    else idealWs.current.load('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');
    
    if (participantUrl) partWs.current.load(participantUrl);
    else partWs.current.load('data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA');

    // Sync play/pause
    idealWs.current.on('play', () => { partWs.current.play(); setIsPlaying(true); });
    idealWs.current.on('pause', () => { partWs.current.pause(); setIsPlaying(false); });
    
    // Sync seeking (bidirectional)
    idealWs.current.on('seek', (prog) => {
      if (partWs.current.getCurrentTime() !== idealWs.current.getCurrentTime()) {
        partWs.current.seekTo(prog);
      }
    });
    partWs.current.on('seek', (prog) => {
      if (idealWs.current.getCurrentTime() !== partWs.current.getCurrentTime()) {
        idealWs.current.seekTo(prog);
      }
    });

    Promise.all([
      new Promise(res => idealWs.current.on('ready', res)),
      new Promise(res => partWs.current.on('ready', res))
    ]).then(() => {
      setIsReady(true);
      
      // Draw regions
      if (regions) {
        regions.forEach(r => {
          wsRegions.addRegion({
            start: r.start,
            end: r.end,
            color: FLAW_COLORS[r.type] || 'rgba(255, 255, 255, 0.2)',
            drag: false,
            resize: false,
            content: r.type.replace('_', ' ')
          });
        });
      }
    });

    return () => {
      idealWs.current?.destroy();
      partWs.current?.destroy();
    };
  }, [idealUrl, participantUrl, regions]);

  // Handle external jumpTime prop
  useEffect(() => {
    if (jumpTime !== null && idealWs.current && isReady) {
      const duration = idealWs.current.getDuration();
      if (duration > 0) {
        idealWs.current.seekTo(jumpTime / duration);
      }
    }
  }, [jumpTime, isReady]);

  const togglePlay = () => {
    if (idealWs.current) {
      idealWs.current.isPlaying() ? idealWs.current.pause() : idealWs.current.play();
    }
  };

  const stopPlay = () => {
    if (idealWs.current) {
      idealWs.current.stop();
      idealWs.current.seekTo(0);
    }
  };

  return (
    <div className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h2>Waveform Analysis</h2>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="demo-btn" onClick={togglePlay} disabled={!isReady} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
            {isPlaying ? 'Pause' : 'Play Both'}
          </button>
          <button className="demo-btn" onClick={stopPlay} disabled={!isReady}>
            <Square size={18} />
          </button>
        </div>
      </div>

      <div className="wave-container">
        <div className="wave-label">Ideal Reference</div>
        <div ref={idealContainerRef}></div>
        <div ref={idealTimelineRef} style={{ height: '20px', marginTop: '5px' }}></div>
      </div>

      <div className="wave-container">
        <div className="wave-label">Participant Recording</div>
        <div ref={partContainerRef}></div>
        <div ref={partTimelineRef} style={{ height: '20px', marginTop: '5px' }}></div>
      </div>
    </div>
  );
};

export default DualWaveform;
