import React from 'react';
import Plot from 'react-plotly.js';

const FeatureTimelines = ({ timelines, regions }) => {
  if (!timelines || !timelines.time) return null;

  const time = timelines.time;

  const createShapes = () => {
    if (!regions) return [];
    return regions.map(r => ({
      type: 'rect',
      xref: 'x',
      yref: 'paper',
      x0: r.start,
      x1: r.end,
      y0: 0,
      y1: 1,
      fillcolor: 'rgba(255, 107, 107, 0.2)', // generic red tint, could map to flaw colors
      line: { width: 0 }
    }));
  };

  const layoutBase = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#c9d1d9' },
    margin: { t: 30, r: 20, l: 50, b: 30 },
    height: 200,
    xaxis: { 
      color: '#30363d',
      gridcolor: '#30363d',
      zerolinecolor: '#30363d'
    },
    yaxis: {
      color: '#30363d',
      gridcolor: '#30363d',
      zerolinecolor: '#30363d'
    },
    shapes: createShapes(),
    showlegend: true,
    legend: { orientation: 'h', y: 1.1 }
  };

  const features = [
    { key: 'rate', title: 'Speech Rate (syllables/sec)' },
    { key: 'pitch', title: 'Pitch (F0 Hz)' },
    { key: 'energy', title: 'RMS Energy' }
  ];

  return (
    <div className="panel">
      <h2>Acoustic Feature Timelines</h2>
      
      {features.map(feat => {
        if (!timelines.ideal[feat.key]) return null;
        
        return (
          <div key={feat.key} style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: 0, color: 'var(--text-muted)' }}>{feat.title}</h3>
            <Plot
              data={[
                {
                  x: time,
                  y: timelines.ideal[feat.key],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Ideal',
                  line: { color: '#4d96ff', width: 2 }
                },
                {
                  x: time,
                  y: timelines.participant[feat.key],
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Participant',
                  line: { color: '#ff6b6b', width: 2 }
                }
              ]}
              layout={{ ...layoutBase }}
              useResizeHandler={true}
              style={{ width: '100%', height: '200px' }}
              config={{ displayModeBar: false }}
            />
          </div>
        );
      })}
    </div>
  );
};

export default FeatureTimelines;
