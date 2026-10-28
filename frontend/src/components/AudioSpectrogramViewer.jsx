import React from 'react';

export function AudioSpectrogramViewer({ audioData, spectrogramUrl }) {
  if (!audioData) {
    return (
      <div className="empty-state" style={{ padding: '24px 0' }}>
        <div className="empty-icon">🎵</div>
        <h3>No Audio Evidence Available</h3>
        <p>Acoustic extraction and log-mel spectrogram data will appear here once processed.</p>
      </div>
    );
  }

  const {
    audio_available,
    has_audio_stream,
    is_silent,
    duration_seconds,
    sample_rate_hz,
    mel_bins,
    time_steps,
    waveform_stats,
    warnings
  } = audioData;

  const rmsEnergy = waveform_stats?.rms_energy ?? (waveform_stats?.rms ?? null);

  return (
    <div className="audio-spectrogram-viewer" style={{ marginTop: '10px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '16px' }}>
        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Audio Stream</div>
          <div style={{ marginTop: '4px' }}>
            {has_audio_stream ? (
              <span className="badge badge-normal">✓ Stream Present</span>
            ) : (
              <span className="badge badge-fire">✕ No Audio Stream</span>
            )}
          </div>
        </div>

        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Acoustic State</div>
          <div style={{ marginTop: '4px' }}>
            {is_silent ? (
              <span className="badge badge-accident">Silent Audio</span>
            ) : audio_available ? (
              <span className="badge badge-normal">Active Signal</span>
            ) : (
              <span className="badge badge-neutral">Unavailable</span>
            )}
          </div>
        </div>

        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Sample Rate / Channels</div>
          <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: '600', color: 'var(--text-main)', marginTop: '4px' }}>
            {sample_rate_hz ? `${sample_rate_hz} Hz · Mono` : '16,000 Hz · Mono'}
          </div>
        </div>

        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Audio Duration</div>
          <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: '600', color: 'var(--text-main)', marginTop: '4px' }}>
            {duration_seconds ? `${Number(duration_seconds).toFixed(2)}s` : '—'}
          </div>
        </div>

        <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Mel Bins × Time Steps</div>
          <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: '600', color: 'var(--text-main)', marginTop: '4px' }}>
            {mel_bins || 64} × {time_steps || 313}
          </div>
        </div>

        {rmsEnergy !== null && (
          <div style={{ background: 'var(--bg-surface)', padding: '12px', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>RMS Acoustic Energy</div>
            <div className="font-mono" style={{ fontSize: '13.5px', fontWeight: '600', color: 'var(--text-main)', marginTop: '4px' }}>
              {Number(rmsEnergy).toFixed(5)}
            </div>
          </div>
        )}
      </div>

      {warnings && warnings.length > 0 && (
        <div style={{ marginBottom: '14px' }}>
          {warnings.map((w, i) => (
            <div key={i} className="disclaimer-banner" style={{ background: 'rgba(59, 130, 246, 0.1)', borderColor: 'rgba(59, 130, 246, 0.3)', color: '#60a5fa' }}>
              <span>ℹ️</span> <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {spectrogramUrl && (
        <div style={{ background: 'var(--bg-surface)', padding: '14px', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
          <div style={{ fontSize: '12.5px', fontWeight: '600', color: 'var(--text-main)', marginBottom: '8px', textAlign: 'left' }}>
            Genuine Log-Mel Spectrogram (64 Mel Bins, 16 kHz Resampled)
          </div>
          <img
            src={spectrogramUrl}
            alt="Extracted Log-Mel Spectrogram"
            style={{ width: '100%', maxHeight: '280px', objectFit: 'contain', borderRadius: 'var(--radius-xs)', background: '#000' }}
          />
        </div>
      )}
    </div>
  );
}
