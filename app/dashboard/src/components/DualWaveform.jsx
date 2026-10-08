import React, { useEffect, useRef, useState, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.esm.js';
import { Play, Pause, Square, Clock, AlertCircle } from 'lucide-react';
import WordAlignmentStrip from './WordAlignmentStrip';

const FLAW_ANNOTATION_COLORS = {
  rushed_pace: 'rgba(201, 111, 74, 0.18)',
  dead_pause: 'rgba(224, 166, 68, 0.20)',
  flat_pitch: 'rgba(92, 124, 88, 0.20)',
  mumbled_clarity: 'rgba(74, 112, 160, 0.18)',
};

const DualWaveform = ({
  idealUrl,
  participantUrl,
  regions = [],
  activePlayRegion = null,
  onRegionSelected = null,
}) => {
  const idealContainerRef = useRef(null);
  const partContainerRef = useRef(null);
  const idealTimelineRef = useRef(null);
  const partTimelineRef = useRef(null);

  const idealWs = useRef(null);
  const partWs = useRef(null);
  const regionsPluginRef = useRef(null);
  const regionStopTimerRef = useRef(null);

  const [playbackMode, setPlaybackMode] = useState('none'); // 'none' | 'reference' | 'participant' | 'both'
  const [isReady, setIsReady] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [audioError, setAudioError] = useState(null);

  const playbackModeRef = useRef('none');
  const isMountedRef = useRef(true);

  const updatePlaybackMode = useCallback((mode) => {
    playbackModeRef.current = mode;
    if (isMountedRef.current) {
      setPlaybackMode(mode);
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Suppress dev-mode AbortError in React StrictMode
  useEffect(() => {
    const handleAbort = (e) => {
      const msg = e?.reason?.message || e?.message || '';
      const name = e?.reason?.name || e?.name || '';
      if (
        name === 'AbortError' ||
        msg.includes('aborted') ||
        msg.includes('Fetch is aborted')
      ) {
        if (e.preventDefault) e.preventDefault();
        return true;
      }
    };
    window.addEventListener('unhandledrejection', handleAbort);
    window.addEventListener('error', handleAbort);
    return () => {
      window.removeEventListener('unhandledrejection', handleAbort);
      window.removeEventListener('error', handleAbort);
    };
  }, []);

  const clearRegionTimer = () => {
    if (regionStopTimerRef.current) {
      clearTimeout(regionStopTimerRef.current);
      regionStopTimerRef.current = null;
    }
  };

  useEffect(() => {
    // If either audio URL is missing, show clear UI state and do not initialize WaveSurfer loading
    if (!idealUrl || !participantUrl) {
      setAudioError('Demo audio unavailable');
      setIsReady(false);
      return;
    }

    setAudioError(null);
    setIsReady(false);

    if (!idealContainerRef.current || !partContainerRef.current) return;

    let isMounted = true;

    const oldIdeal = idealWs.current;
    const oldPart = partWs.current;
    idealWs.current = null;
    partWs.current = null;

    if (oldIdeal) { try { oldIdeal.pause(); oldIdeal.destroy(); } catch (e) {} }
    if (oldPart) { try { oldPart.pause(); oldPart.destroy(); } catch (e) {} }

    // REFERENCE: neutral warm slate tone
    const wsIdeal = WaveSurfer.create({
      container: idealContainerRef.current,
      waveColor: '#A0AEC0',
      progressColor: '#4A5568',
      cursorColor: '#2D3748',
      cursorWidth: 1.5,
      height: 64,
      normalize: true,
      plugins: [
        TimelinePlugin.create({
          container: idealTimelineRef.current,
          primaryLabelFontColor: '#777777',
          secondaryLabelFontColor: '#A0AEC0',
          primaryLabelInterval: 1,
        }),
      ],
    });

    // PARTICIPANT: terracotta / coral tone
    const regionsPlugin = RegionsPlugin.create();
    regionsPluginRef.current = regionsPlugin;

    const wsPart = WaveSurfer.create({
      container: partContainerRef.current,
      waveColor: '#FFB8A5',
      progressColor: '#C96F4A',
      cursorColor: '#C96F4A',
      cursorWidth: 1.5,
      height: 64,
      normalize: true,
      plugins: [
        regionsPlugin,
        TimelinePlugin.create({
          container: partTimelineRef.current,
          primaryLabelFontColor: '#777777',
          secondaryLabelFontColor: '#A0AEC0',
          primaryLabelInterval: 1,
        }),
      ],
    });

    idealWs.current = wsIdeal;
    partWs.current = wsPart;

    // Error handling: listen on error and show user-friendly message without throwing runtime errors
    wsIdeal.on('error', (err) => {
      console.warn('Ideal WaveSurfer error:', err);
      if (isMounted) setAudioError('Unable to decode demo audio.');
    });

    wsPart.on('error', (err) => {
      console.warn('Participant WaveSurfer error:', err);
      if (isMounted) setAudioError('Unable to decode demo audio.');
    });

    // Load actual audio URLs
    wsIdeal.load(idealUrl).catch((err) => {
      console.warn('Ideal load aborted or failed:', err);
    });
    wsPart.load(participantUrl).catch((err) => {
      console.warn('Participant load aborted or failed:', err);
    });

    // Handle audio completion / pause events cleanly
    wsIdeal.on('pause', () => {
      if (!isMounted) return;
      if (playbackModeRef.current === 'reference') {
        updatePlaybackMode('none');
      } else if (playbackModeRef.current === 'both' && (!partWs.current || !partWs.current.isPlaying())) {
        updatePlaybackMode('none');
      }
    });

    wsPart.on('pause', () => {
      if (!isMounted) return;
      if (playbackModeRef.current === 'participant') {
        updatePlaybackMode('none');
      } else if (playbackModeRef.current === 'both' && (!idealWs.current || !idealWs.current.isPlaying())) {
        updatePlaybackMode('none');
      }
    });

    wsIdeal.on('finish', () => {
      if (!isMounted) return;
      if (playbackModeRef.current === 'reference') {
        updatePlaybackMode('none');
      } else if (playbackModeRef.current === 'both' && (!partWs.current || !partWs.current.isPlaying())) {
        updatePlaybackMode('none');
      }
    });

    wsPart.on('finish', () => {
      if (!isMounted) return;
      if (playbackModeRef.current === 'participant') {
        updatePlaybackMode('none');
      } else if (playbackModeRef.current === 'both' && (!idealWs.current || !idealWs.current.isPlaying())) {
        updatePlaybackMode('none');
      }
    });

    // Sync seeking across both waveforms
    let isSeekingSync = false;
    wsIdeal.on('seek', (prog) => {
      if (isSeekingSync) return;
      isSeekingSync = true;
      try {
        if (wsPart && Math.abs(wsPart.getProgress() - prog) > 0.005) wsPart.seekTo(prog);
        const maxDur = Math.max(wsIdeal.getDuration() || 0, wsPart?.getDuration() || 0) || 18.0;
        if (isMounted) setCurrentTime(prog * maxDur);
      } catch (e) {}
      isSeekingSync = false;
    });

    wsPart.on('seek', (prog) => {
      if (isSeekingSync) return;
      isSeekingSync = true;
      try {
        if (wsIdeal && Math.abs(wsIdeal.getProgress() - prog) > 0.005) wsIdeal.seekTo(prog);
        const maxDur = Math.max(wsIdeal?.getDuration() || 0, wsPart.getDuration() || 0) || 18.0;
        if (isMounted) setCurrentTime(prog * maxDur);
      } catch (e) {}
      isSeekingSync = false;
    });

    // Time update listener
    wsIdeal.on('timeupdate', (t) => {
      if (isMounted && playbackModeRef.current === 'reference') {
        setCurrentTime(t);
      }
    });

    wsPart.on('timeupdate', (t) => {
      if (isMounted && playbackModeRef.current !== 'reference') {
        setCurrentTime(t);
      }
    });

    // Readiness: disabled until BOTH audio files emit ready
    let idealReady = false;
    let partReady = false;
    const checkBothReady = () => {
      if (idealReady && partReady && isMounted) {
        setIsReady(true);
        const maxDur = Math.max(wsIdeal.getDuration() || 0, wsPart.getDuration() || 0) || 18.0;
        setDuration(maxDur);

        // Analytical annotations for flaw regions
        if (regions && regions.length > 0) {
          try {
            regionsPlugin.clearRegions();
            regions.forEach((r) => {
              const tint = FLAW_ANNOTATION_COLORS[r.type] || 'rgba(201, 111, 74, 0.16)';
              const regItem = regionsPlugin.addRegion({
                id: r.id,
                start: r.start,
                end: r.end,
                color: tint,
                drag: false,
                resize: false,
                content: `${r.type.replace('_', ' ')} [${r.severity}]`,
              });

              regItem.on('click', () => {
                playSpan(r.start, r.end);
                if (onRegionSelected) onRegionSelected(r);
              });
            });
          } catch (e) {}
        }
      }
    };

    wsIdeal.on('ready', () => { idealReady = true; checkBothReady(); });
    wsPart.on('ready', () => { partReady = true; checkBothReady(); });

    return () => {
      isMounted = false;
      clearRegionTimer();
      try {
        if (wsIdeal) {
          wsIdeal.unAll();
          wsIdeal.pause();
          setTimeout(() => { try { wsIdeal.destroy(); } catch (e) {} }, 0);
        }
        if (wsPart) {
          wsPart.unAll();
          wsPart.pause();
          setTimeout(() => { try { wsPart.destroy(); } catch (e) {} }, 0);
        }
      } catch (e) {}
    };
  }, [idealUrl, participantUrl, regions, updatePlaybackMode]);

  const playSpan = useCallback((start, end) => {
    clearRegionTimer();
    if (!idealWs.current || !partWs.current || !isReady) return;

    const safeDuration = duration || 18.0;
    const progressStart = Math.min(1.0, Math.max(0, start / safeDuration));

    idealWs.current.seekTo(progressStart);
    partWs.current.seekTo(progressStart);
    idealWs.current.setPlaybackRate(playbackSpeed);
    partWs.current.setPlaybackRate(playbackSpeed);
    idealWs.current.play().catch(() => {});
    partWs.current.play().catch(() => {});
    updatePlaybackMode('both');

    const playMs = Math.max(200, (end - start) * 1000 * (1 / playbackSpeed));
    regionStopTimerRef.current = setTimeout(() => {
      if (idealWs.current) idealWs.current.pause();
      if (partWs.current) partWs.current.pause();
      updatePlaybackMode('none');
    }, playMs);
  }, [isReady, duration, playbackSpeed, updatePlaybackMode]);

  useEffect(() => {
    if (activePlayRegion && isReady) {
      playSpan(activePlayRegion.start, activePlayRegion.end);
    }
  }, [activePlayRegion, isReady, playSpan]);

  // PLAY REFERENCE: play ONLY ideal/reference; participant stays stopped; mutual exclusion
  const togglePlayReference = () => {
    clearRegionTimer();
    if (!isReady || !idealWs.current || !partWs.current) return;

    if (playbackModeRef.current === 'reference') {
      idealWs.current.pause();
      updatePlaybackMode('none');
    } else {
      updatePlaybackMode('reference');
      partWs.current.pause();
      idealWs.current.setPlaybackRate(playbackSpeed);
      idealWs.current.play().catch(() => {});
    }
  };

  // PLAY PARTICIPANT: play ONLY participant; reference stays stopped; mutual exclusion
  const togglePlayParticipant = () => {
    clearRegionTimer();
    if (!isReady || !idealWs.current || !partWs.current) return;

    if (playbackModeRef.current === 'participant') {
      partWs.current.pause();
      updatePlaybackMode('none');
    } else {
      updatePlaybackMode('participant');
      idealWs.current.pause();
      partWs.current.setPlaybackRate(playbackSpeed);
      partWs.current.play().catch(() => {});
    }
  };

  // PLAY BOTH: synchronized playback of reference + participant; mutual exclusion
  const togglePlayBoth = () => {
    clearRegionTimer();
    if (!isReady || !idealWs.current || !partWs.current) return;

    if (playbackModeRef.current === 'both') {
      idealWs.current.pause();
      partWs.current.pause();
      updatePlaybackMode('none');
    } else {
      updatePlaybackMode('both');
      idealWs.current.setPlaybackRate(playbackSpeed);
      partWs.current.setPlaybackRate(playbackSpeed);
      idealWs.current.play().catch(() => {});
      partWs.current.play().catch(() => {});
    }
  };

  // STOP: stop ALL audio, rewind to 0, reset playback state
  const stopPlayback = () => {
    clearRegionTimer();
    updatePlaybackMode('none');
    if (idealWs.current) {
      idealWs.current.pause();
      idealWs.current.seekTo(0);
    }
    if (partWs.current) {
      partWs.current.pause();
      partWs.current.seekTo(0);
    }
    setCurrentTime(0);
  };

  // SPEED: applies consistently to Reference, Participant, and Play Both
  const handleSpeedChange = (spd) => {
    setPlaybackSpeed(spd);
    if (idealWs.current) idealWs.current.setPlaybackRate(spd);
    if (partWs.current) partWs.current.setPlaybackRate(spd);
  };

  const formatTime = (secs) => {
    const s = Math.max(0, secs);
    const m = Math.floor(s / 60);
    const rem = (s % 60).toFixed(1);
    return `${m < 10 ? '0' : ''}${m}:${rem < 10 ? '0' : ''}${rem}`;
  };

  return (
    <div className="contrastive-section">
      {/* Top Bar */}
      <div className="contrastive-top-bar">
        <div className="contrastive-title-group">
          <h3 className="contrastive-heading">Contrastive Speech Waveform</h3>
          <span className="contrastive-hint">Word-aligned acoustic comparison</span>
        </div>

        <div className="transport-bar" role="toolbar" aria-label="Audio playback controls">
          {/* Time display on the left */}
          <span className="timecode-readout" aria-label="Playback timestamp">
            <Clock size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
            {formatTime(currentTime)} / {formatTime(duration || 18.0)}
          </span>

          {/* Compact premium control group: [▶ Reference] [▶ Participant] [▶ Play Both] [■] [1x] [1.25x] */}
          <div className="playback-controls-group">
            {/* 1. Play Reference */}
            <button
              className={`btn-playback-control btn-playback-reference ${playbackMode === 'reference' ? 'is-active is-playing' : ''}`}
              onClick={togglePlayReference}
              disabled={!isReady}
              title={playbackMode === 'reference' ? 'Pause reference audio' : 'Play reference audio'}
              aria-label="Play reference audio"
              aria-pressed={playbackMode === 'reference'}
            >
              {playbackMode === 'reference' ? (
                <Pause size={13} />
              ) : (
                <Play size={13} fill="currentColor" />
              )}
              <span>{playbackMode === 'reference' ? 'Pause' : 'Reference'}</span>
              {playbackMode === 'reference' && <span className="sr-only"> Reference</span>}
            </button>

            {/* 2. Play Participant */}
            <button
              className={`btn-playback-control btn-playback-participant ${playbackMode === 'participant' ? 'is-active is-playing' : ''}`}
              onClick={togglePlayParticipant}
              disabled={!isReady}
              title={playbackMode === 'participant' ? 'Pause participant audio' : 'Play participant audio'}
              aria-label="Play participant audio"
              aria-pressed={playbackMode === 'participant'}
            >
              {playbackMode === 'participant' ? (
                <Pause size={13} />
              ) : (
                <Play size={13} fill="currentColor" />
              )}
              <span>{playbackMode === 'participant' ? 'Pause' : 'Participant'}</span>
              {playbackMode === 'participant' && <span className="sr-only"> Participant</span>}
            </button>

            {/* 3. Play Both */}
            <button
              className={`btn-playback-control btn-playback-both ${playbackMode === 'both' ? 'is-active is-playing' : ''}`}
              onClick={togglePlayBoth}
              disabled={!isReady}
              title={playbackMode === 'both' ? 'Pause synchronized playback' : 'Play both synchronized'}
              aria-label="Play both synchronized"
              aria-pressed={playbackMode === 'both'}
            >
              {playbackMode === 'both' ? (
                <Pause size={13} />
              ) : (
                <Play size={13} fill="currentColor" />
              )}
              <span>{playbackMode === 'both' ? 'Pause' : 'Play Both'}</span>
              {playbackMode === 'both' && <span className="sr-only"> Play Both</span>}
            </button>

            {/* 4. Stop Button */}
            <button
              className="btn-playback-control btn-playback-stop"
              onClick={stopPlayback}
              disabled={!isReady}
              title="Stop playback"
              aria-label="Stop playback"
            >
              <Square size={11} fill="currentColor" />
            </button>

            {/* 5. Speed Controls: 1x, 1.25x */}
            <div className="playback-speed-group" role="group" aria-label="Playback speed selection">
              {[
                { value: 1.0, label: '1x', titleText: 'Playback speed 1x' },
                { value: 1.25, label: '1.25x', titleText: 'Playback speed 1.25x' },
              ].map(({ value, label, titleText }) => (
                <button
                  key={value}
                  className={`btn-playback-speed ${playbackSpeed === value ? 'is-active' : ''}`}
                  onClick={() => handleSpeedChange(value)}
                  disabled={!isReady}
                  title={titleText}
                  aria-label={titleText}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* User-friendly audio error notice if triggered */}
      {audioError && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 14px',
            marginBottom: '16px',
            backgroundColor: 'var(--bg-peach-subtle)',
            border: '1px solid var(--accent-peach)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--accent-terracotta)',
            fontSize: '0.85rem',
          }}
        >
          <AlertCircle size={16} />
          <span>{audioError}</span>
        </div>
      )}

      {/* Waveform Channels */}
      <div className="waveform-track-container">
        {/* Track 1: REFERENCE */}
        <div className="waveform-channel-card channel-reference">
          <div className="channel-tag-line">
            <span className="tag-reference">REFERENCE</span>
            <span style={{ fontSize: '0.72rem', color: '#777777', textTransform: 'none' }}>Ideal Baseline Take</span>
          </div>
          <div ref={idealContainerRef} className="waveform-canvas-box" />
          <div ref={idealTimelineRef} className="waveform-timeline-box" />
        </div>

        {/* Track 2: PARTICIPANT */}
        <div className="waveform-channel-card channel-participant">
          <div className="channel-tag-line">
            <span className="tag-participant">participant</span>
            <span style={{ fontSize: '0.72rem', color: '#C96F4A', textTransform: 'none' }}>Candidate Take (Flaws Annotated)</span>
          </div>
          <div ref={partContainerRef} className="waveform-canvas-box" />
          <div ref={partTimelineRef} className="waveform-timeline-box" />
        </div>
      </div>

      {/* Word Alignment Strip */}
      <WordAlignmentStrip
        activeTime={currentTime}
        onWordClick={(start, end) => playSpan(start, end)}
      />
    </div>
  );
};

export default DualWaveform;
