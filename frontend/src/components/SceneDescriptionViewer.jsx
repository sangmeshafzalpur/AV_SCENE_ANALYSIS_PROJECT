import React from 'react';

export function SceneDescriptionViewer({ descriptionData, formattedText }) {
  if (!descriptionData && !formattedText) {
    return (
      <div className="empty-state" style={{ padding: '24px 0' }}>
        <div className="empty-icon">📝</div>
        <h3>No Scene Description Available</h3>
        <p>Evidence-grounded structured scene description will appear here after video analysis.</p>
      </div>
    );
  }

  const structured = descriptionData?.structured || descriptionData || {};
  const {
    visual_observations,
    audio_observations,
    speech_transcript_and_summary,
    key_timestamps,
    model_prediction_and_class_scores,
    uncertainty_and_limitations
  } = structured;

  return (
    <div className="scene-description-viewer" style={{ marginTop: '10px' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Section 1: Visual Observations */}
        <div className="glass-card" style={{ padding: '16px' }}>
          <div className="card-title" style={{ fontSize: '14px', marginBottom: '8px', color: '#60a5fa' }}>
            <span>👁️ 1. Visual Observations</span>
          </div>
          {visual_observations ? (
            <div style={{ fontSize: '13px', lineHeight: 1.6 }}>
              <div>• Sampled Frames: <strong>{visual_observations.sampled_frame_count ?? 12}</strong> frames</div>
              <div>• Video Duration: <strong>{Number(visual_observations.video_duration_sec ?? 0).toFixed(2)}s</strong> ({visual_observations.resolution || '224×224 normalized'})</div>
              {visual_observations.key_influential_frames && visual_observations.key_influential_frames.length > 0 && (
                <div style={{ marginTop: '6px' }}>
                  • Key Influential Frames:
                  <ul style={{ paddingLeft: '20px', marginTop: '4px' }}>
                    {visual_observations.key_influential_frames.map((k, i) => (
                      <li key={i}>
                        Rank #{k.rank}: Frame {k.frame_index + 1} at {Number(k.timestamp_sec).toFixed(2)}s (drop: {(Number(k.score_drop_delta) * 100).toFixed(2)}%)
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Visual evidence not available</div>
          )}
        </div>

        {/* Section 2: Audio Observations */}
        <div className="glass-card" style={{ padding: '16px' }}>
          <div className="card-title" style={{ fontSize: '14px', marginBottom: '8px', color: '#34d399' }}>
            <span>🎧 2. Audio Observations</span>
          </div>
          {audio_observations ? (
            <div style={{ fontSize: '13px', lineHeight: 1.6 }}>
              <div>• Audio Presence: <strong>{audio_observations.has_audio_stream ? 'Present' : 'Absent'}</strong> (State: <em>{audio_observations.acoustic_state || 'unknown'}</em>)</div>
              <div>• Mel Spectrogram: <strong>{audio_observations.mel_bins ?? 64} bins × {audio_observations.time_steps ?? 313} steps</strong> (16 kHz resampled)</div>
              {audio_observations.rms_energy !== undefined && (
                <div>• Measured RMS Energy: <strong>{Number(audio_observations.rms_energy).toFixed(5)}</strong></div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Audio evidence not available</div>
          )}
        </div>

        {/* Section 3: Speech Transcript and Summary */}
        <div className="glass-card" style={{ padding: '16px' }}>
          <div className="card-title" style={{ fontSize: '14px', marginBottom: '8px', color: '#a78bfa' }}>
            <span>🗣️ 3. Speech Transcript and Summary</span>
          </div>
          {speech_transcript_and_summary ? (
            <div style={{ fontSize: '13px', lineHeight: 1.6 }}>
              <div>• Language: <strong>{speech_transcript_and_summary.detected_language === 'kannada' ? 'ಕನ್ನಡ (Kannada)' : speech_transcript_and_summary.detected_language === 'english' ? 'English' : 'None / Silent'}</strong></div>
              <div style={{ marginTop: '6px' }}>
                • Grounded Summary: <em>{speech_transcript_and_summary.grounded_summary}</em>
              </div>
              {speech_transcript_and_summary.raw_transcript && (
                <div className="kannada-text" style={{ marginTop: '8px', background: 'var(--bg-base)', padding: '8px 12px', borderRadius: '4px' }}>
                  "{speech_transcript_and_summary.raw_transcript}"
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Speech evidence not available</div>
          )}
        </div>

        {/* Section 4: Key Timestamps */}
        <div className="glass-card" style={{ padding: '16px' }}>
          <div className="card-title" style={{ fontSize: '14px', marginBottom: '8px', color: '#fbbf24' }}>
            <span>⏱️ 4. Key Timestamps</span>
          </div>
          {key_timestamps && key_timestamps.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '13px' }}>
              {key_timestamps.map((t, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span className="font-mono badge badge-neutral">{Number(t.timestamp_sec).toFixed(2)}s</span>
                  <span>{t.event}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Key timeline markers not available</div>
          )}
        </div>

        {/* Section 5: Model Prediction and Class Scores */}
        <div className="glass-card" style={{ padding: '16px' }}>
          <div className="card-title" style={{ fontSize: '14px', marginBottom: '8px', color: '#ff7074' }}>
            <span>🎯 5. Model Prediction and Class Scores</span>
          </div>
          {model_prediction_and_class_scores ? (
            <div style={{ fontSize: '13px', lineHeight: 1.6 }}>
              <div>• Predicted Class: <strong>{model_prediction_and_class_scores.predicted_class}</strong></div>
              <div>• Top Softmax Activation: <strong>{(Number(model_prediction_and_class_scores.confidence) * 100).toFixed(2)}%</strong></div>
              {model_prediction_and_class_scores.class_scores && (
                <div style={{ marginTop: '6px' }}>
                  • Score Distribution: Fire: {(Number(model_prediction_and_class_scores.class_scores.Fire || 0) * 100).toFixed(2)}% | Accident: {(Number(model_prediction_and_class_scores.class_scores.Accident || 0) * 100).toFixed(2)}% | Normal: {(Number(model_prediction_and_class_scores.class_scores.Normal || 0) * 100).toFixed(2)}%
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Prediction scores not available</div>
          )}
        </div>

        {/* Section 6: Uncertainty and Limitations */}
        <div className="glass-card" style={{ padding: '16px', borderLeft: '3px solid #F59E0B' }}>
          <div className="card-title" style={{ fontSize: '14px', marginBottom: '8px', color: '#fbbf24' }}>
            <span>⚖️ 6. Uncertainty and Limitations</span>
          </div>
          {uncertainty_and_limitations ? (
            <div style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              <div>• <strong>Causation Disclaimer:</strong> {uncertainty_and_limitations.causation_disclaimer}</div>
              <div style={{ marginTop: '4px' }}>• <strong>Calibration:</strong> {uncertainty_and_limitations.calibration_statement}</div>
              <div style={{ marginTop: '4px' }}>• <strong>Boundaries:</strong> {uncertainty_and_limitations.evidence_boundary_statement}</div>
            </div>
          ) : (
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Uncertainty statement not available</div>
          )}
        </div>
      </div>
    </div>
  );
}
