import React from 'react';
import { api, BASE_URL } from '../api';

export function FrameViewer({ frames, attributions, filmstripUrl }) {
  if (!frames || frames.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '24px 0' }}>
        <div className="empty-icon">🎞️</div>
        <h3>No Frames Available</h3>
        <p>12 extracted frames will appear here once video analysis completes.</p>
      </div>
    );
  }

  // Map attributions by frame index for quick lookup
  const attrMap = {};
  if (attributions && Array.isArray(attributions)) {
    attributions.forEach(a => {
      attrMap[a.frame_index] = a;
    });
  }

  return (
    <div className="frame-viewer-container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          Extracted 12 temporally equidistant frames (224×224 RGB)
        </div>
        {filmstripUrl && (
          <a
            href={filmstripUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-secondary btn-sm"
          >
            🖼️ View Filmstrip
          </a>
        )}
      </div>

      <div className="frames-grid" id="frames-grid-viewer">
        {frames.map((frame, idx) => {
          const attr = attrMap[idx];
          const rank = attr?.rank;
          const rankClass = rank === 1 ? 'rank-1' : rank === 2 ? 'rank-2' : rank === 3 ? 'rank-3' : '';
          const rawUrl = frame.download_url || (frame.artifact_id ? api.getArtifactUrl(frame.job_id, frame.artifact_id) : null);
          const frameUrl = rawUrl ? (rawUrl.startsWith('http') ? rawUrl : `${BASE_URL}${rawUrl.startsWith('/') ? '' : '/'}${rawUrl}`) : null;
          const timestampSec = frame.metadata?.timestamp_sec ?? (attr?.timestamp_sec ?? (idx * 0.5));

          return (
            <div className={`frame-card ${rankClass}`} key={frame.artifact_id || idx} id={`frame-card-${idx}`}>
              {rank && rank <= 3 && (
                <div className={`frame-rank-badge rank-${rank}`}>
                  #{rank} Influential
                </div>
              )}
              {frameUrl ? (
                <img
                  src={frameUrl}
                  alt={`Frame ${idx + 1} at ${timestampSec.toFixed(2)}s`}
                  className="frame-img"
                  loading="lazy"
                />
              ) : (
                <div className="frame-img" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                  Frame {idx + 1}
                </div>
              )}
              <div className="frame-meta">
                <span className="font-mono" style={{ fontWeight: '600' }}>#{idx + 1}</span>
                <span className="font-mono">{Number(timestampSec).toFixed(2)}s</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
