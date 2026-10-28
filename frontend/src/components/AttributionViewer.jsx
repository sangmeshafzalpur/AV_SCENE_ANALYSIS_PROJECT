import React from 'react';

export function AttributionViewer({ attributionsData }) {
  if (!attributionsData || !attributionsData.attributions || attributionsData.attributions.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '24px 0' }}>
        <div className="empty-icon">🎯</div>
        <h3>No Frame Attributions Available</h3>
        <p>Perturbation-based influential frame analysis requires an executed video analysis job.</p>
      </div>
    );
  }

  const {
    target_class,
    baseline_score,
    attribution_method,
    causation_disclaimer,
    attributions
  } = attributionsData;

  // Sort by rank ascending
  const sortedAttributions = [...attributions].sort((a, b) => a.rank - b.rank);

  return (
    <div className="attribution-viewer" style={{ marginTop: '10px' }}>
      <div className="disclaimer-banner">
        <span style={{ fontSize: '18px' }}>⚠️</span>
        <div>
          <strong>Model Sensitivity Disclaimer:</strong>{' '}
          {causation_disclaimer || 'Perturbation results indicate model sensitivity, not proof of real-world causation.'}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', margin: '14px 0' }}>
        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Target Class</div>
          <div style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-main)', marginTop: '2px' }}>
            {target_class}
          </div>
        </div>
        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Baseline Score</div>
          <div style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-main)', marginTop: '2px' }}>
            {(baseline_score * 100).toFixed(2)}%
          </div>
        </div>
        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Method</div>
          <div style={{ fontSize: '12.5px', fontWeight: '600', color: 'var(--text-body)', marginTop: '2px' }}>
            {attribution_method || 'Temporal Frame Ablation'}
          </div>
        </div>
      </div>

      <div className="table-responsive">
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Frame #</th>
              <th>Timestamp</th>
              <th>Baseline</th>
              <th>Perturbed</th>
              <th>Score Drop (Δ)</th>
              <th>Sensitivity Visual</th>
            </tr>
          </thead>
          <tbody>
            {sortedAttributions.map((a) => {
              const scoreDrop = a.score_difference ?? (a.baseline_score - a.perturbed_score);
              const dropPercent = Math.max(0, scoreDrop * 100);

              return (
                <tr key={a.frame_index}>
                  <td>
                    {a.rank <= 3 ? (
                      <span className={`badge ${a.rank === 1 ? 'badge-fire' : a.rank === 2 ? 'badge-accident' : 'badge-info'}`}>
                        #{a.rank}
                      </span>
                    ) : (
                      <span className="font-mono" style={{ color: 'var(--text-muted)' }}>#{a.rank}</span>
                    )}
                  </td>
                  <td className="font-mono">Frame {a.frame_index + 1}</td>
                  <td className="font-mono">{Number(a.timestamp_sec).toFixed(2)}s</td>
                  <td className="font-mono">{(a.baseline_score * 100).toFixed(2)}%</td>
                  <td className="font-mono">{(a.perturbed_score * 100).toFixed(2)}%</td>
                  <td className="font-mono" style={{ color: scoreDrop > 0.05 ? '#ff6b70' : 'var(--text-main)', fontWeight: scoreDrop > 0.05 ? '700' : '400' }}>
                    {scoreDrop > 0 ? `-${(scoreDrop * 100).toFixed(2)}%` : `+${(Math.abs(scoreDrop) * 100).toFixed(2)}%`}
                  </td>
                  <td style={{ minWidth: '120px' }}>
                    <div className="prob-bar-track" style={{ height: '6px' }}>
                      <div
                        className="prob-bar-fill fire"
                        style={{ width: `${Math.min(dropPercent * 4, 100)}%` }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
