import React from 'react';

export function TranscriptViewer({ transcripts, rawTranscript }) {
  const hasTranscripts = transcripts && Array.isArray(transcripts) && transcripts.length > 0;
  const singleTranscript = hasTranscripts ? transcripts[0] : null;
  const transcriptText = singleTranscript?.transcript || rawTranscript || '';
  const language = singleTranscript?.language || (transcriptText ? 'auto' : 'none');
  const chunks = singleTranscript?.chunks || [];
  const isAccepted = singleTranscript?.is_accepted ?? (transcriptText ? true : false);

  if (!transcriptText && (!chunks || chunks.length === 0)) {
    return (
      <div className="empty-state" style={{ padding: '24px 0' }}>
        <div className="empty-icon">🎙️</div>
        <h3>No Speech Detected</h3>
        <p>No valid speech was detected in this video, or audio was silent. The multimodal model accommodates empty transcripts without hallucinating.</p>
      </div>
    );
  }

  const isKannada = /[\u0C80-\u0CFF]/.test(transcriptText);

  return (
    <div className="transcript-viewer" style={{ marginTop: '10px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className={`badge ${isKannada ? 'badge-purple' : 'badge-info'}`}>
            {isKannada ? 'ಕನ್ನಡ (kn)' : `English (${language})`}
          </span>
          {isAccepted ? (
            <span className="badge badge-normal">✓ Quality Passed</span>
          ) : (
            <span className="badge badge-accident">Filtered</span>
          )}
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Model: vasista22/whisper-kannada-small
        </div>
      </div>

      <div
        className="kannada-transcript-card"
        style={{
          background: 'var(--bg-surface)',
          padding: '16px 20px',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-subtle)',
          marginBottom: '14px'
        }}
      >
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>
          Complete Verified Transcript
        </div>
        <div
          className={`kannada-text ${isKannada ? 'kannada-font' : ''}`}
          style={{
            fontSize: isKannada ? '17px' : '15px',
            color: 'var(--text-main)',
            lineHeight: 1.8,
            wordBreak: 'break-word'
          }}
        >
          {transcriptText}
        </div>
      </div>

      {chunks && chunks.length > 0 && (
        <div style={{ marginTop: '14px' }}>
          <div style={{ fontSize: '12.5px', fontWeight: '600', color: 'var(--text-main)', marginBottom: '8px' }}>
            Timestamped Speech Segments
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {chunks.map((chunk, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                  background: 'var(--bg-surface)',
                  padding: '10px 14px',
                  borderRadius: 'var(--radius-xs)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                <span
                  className="font-mono badge badge-neutral"
                  style={{ whiteSpace: 'nowrap', marginTop: '2px' }}
                >
                  {Number(chunk.start_sec).toFixed(2)}s - {Number(chunk.end_sec).toFixed(2)}s
                </span>
                <span
                  className={`kannada-text ${isKannada ? 'kannada-font' : ''}`}
                  style={{ fontSize: isKannada ? '14.5px' : '13.5px', color: 'var(--text-main)' }}
                >
                  {chunk.text}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
