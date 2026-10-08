import React from 'react';
import Plot from 'react-plotly.js';

const FeatureTimelines = ({ timelines, regions = [] }) => {
  if (!timelines || !timelines.time) return null;

  const time = timelines.time;

  // Analytical flaw overlays with subtle terracotta/peach tint
  const flawShapes = (regions || []).map((r) => ({
    type: 'rect',
    xref: 'x',
    yref: 'paper',
    x0: r.start,
    x1: r.end,
    y0: 0,
    y1: 1,
    fillcolor: 'rgba(201, 111, 74, 0.12)',
    line: {
      color: '#FFCDB8',
      width: 1,
      dash: 'dot',
    },
  }));

  const compactLayout = {
    paper_bgcolor: '#FFFFFF',
    plot_bgcolor: '#FFFDF7',
    font: {
      family: 'Inter, sans-serif',
      color: '#777777',
      size: 10,
    },
    margin: { t: 8, r: 12, l: 44, b: 22 },
    height: 120,
    autosize: true,
    hovermode: 'x unified',
    xaxis: {
      color: '#777777',
      gridcolor: '#F4ECE4',
      zerolinecolor: '#E8DDD2',
      tickformat: '.1f',
      ticksuffix: 's',
      showspikes: true,
      spikethickness: 1,
      spikedash: 'dot',
      spikecolor: '#C96F4A',
    },
    yaxis: {
      color: '#777777',
      gridcolor: '#F4ECE4',
      zerolinecolor: '#E8DDD2',
    },
    shapes: flawShapes,
    showlegend: true,
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.35,
      font: { color: '#252525', size: 9 },
      bgcolor: 'transparent',
    },
  };

  const chartSeries = [
    {
      id: 'rate',
      title: 'Speech Rate',
      unit: 'syll/s',
      ideal: timelines.ideal.rate,
      participant: timelines.participant.rate,
    },
    {
      id: 'pitch',
      title: 'Pitch (F0)',
      unit: 'Hz',
      ideal: timelines.ideal.pitch,
      participant: timelines.participant.pitch,
    },
    {
      id: 'energy',
      title: 'RMS Energy',
      unit: 'RMS',
      ideal: timelines.ideal.energy,
      participant: timelines.participant.energy,
    },
  ];

  return (
    <div className="acoustic-signals-module">
      <div className="signals-header-block">
        <div className="signals-eyebrow">Continuous Signals</div>
        <h3 className="signals-main-title">ACOUSTIC SIGNALS</h3>

        {/* Section 6 Summary Strip */}
        <div className="signals-summary-strip">
          <div className="signal-metric-pill">
            <span className="metric-pill-label">Speech Rate</span>
            <span className="metric-pill-val is-delta-elevated">+45%</span>
          </div>
          <div className="signal-metric-pill">
            <span className="metric-pill-label">Pitch</span>
            <span className="metric-pill-val is-delta-elevated">-76%</span>
          </div>
          <div className="signal-metric-pill">
            <span className="metric-pill-label">Energy</span>
            <span className="metric-pill-val">stable</span>
          </div>
        </div>
      </div>

      {/* 3 Unified Compact Plotly Charts */}
      {chartSeries.map((series) => {
        if (!series.ideal || !series.participant) return null;

        return (
          <div key={series.id} className="compact-chart-box">
            <div className="chart-meta-row">
              <span className="chart-metric-title">{series.title}</span>
              <span className="chart-subtle-hint">{series.unit}</span>
            </div>

            <Plot
              data={[
                {
                  x: time,
                  y: series.ideal,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Reference',
                  line: { color: '#4A5568', width: 1.8 },
                  hovertemplate: `Ref: %{y:.2f}<extra></extra>`,
                },
                {
                  x: time,
                  y: series.participant,
                  type: 'scatter',
                  mode: 'lines',
                  name: 'Participant',
                  line: { color: '#C96F4A', width: 1.8 },
                  hovertemplate: `Part: %{y:.2f}<extra></extra>`,
                },
              ]}
              layout={{
                ...compactLayout,
                yaxis: {
                  ...compactLayout.yaxis,
                  ticksuffix: ` ${series.unit}`,
                },
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '120px' }}
              config={{
                displayModeBar: false,
                responsive: true,
              }}
            />
          </div>
        );
      })}
    </div>
  );
};

export default FeatureTimelines;
