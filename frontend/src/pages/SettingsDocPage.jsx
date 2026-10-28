import React, { useState } from 'react';

export function SettingsDocPage() {
  const [activeTab, setActiveTab] = useState('docs'); // 'docs' | 'api' | 'privacy' | 'troubleshooting'

  const apiEndpoints = [
    { method: 'GET', path: '/health', desc: 'Liveness and database/model readiness check' },
    { method: 'GET', path: '/api/model/metadata', desc: 'Verified model parameters, layers, checksum, and tensor shapes' },
    { method: 'POST', path: '/api/jobs/analyze', desc: 'Asynchronous video analysis job submission' },
    { method: 'GET', path: '/api/jobs/{job_id}', desc: 'Fetch job status, progress, prediction, and evidence metadata' },
    { method: 'GET', path: '/api/jobs/{job_id}/attributions', desc: 'Influential frame perturbation sensitivity results' },
    { method: 'GET', path: '/api/jobs/{job_id}/transcripts', desc: 'Separate speech transcript records with chunk timestamps' },
    { method: 'GET', path: '/api/jobs/{job_id}/description', desc: '6-section evidence-grounded structured scene description' },
    { method: 'GET', path: '/api/jobs/{job_id}/frames', desc: 'List extracted 12 frame artifacts and thumbnail references' },
    { method: 'GET', path: '/api/jobs/{job_id}/audio', desc: 'Waveform statistics and log-mel spectrogram metadata' },
    { method: 'GET', path: '/api/jobs/{job_id}/report', desc: 'Download comprehensive report in HTML, JSON, or Markdown format' },
    { method: 'GET', path: '/api/jobs', desc: 'Searchable analysis history with status/prediction filters and pagination' },
    { method: 'GET', path: '/api/evaluation/metadata', desc: 'Verified 270 blind test metrics, confusion matrix, and splits' },
    { method: 'GET', path: '/api/system/stats', desc: 'Operational telemetry, memory RSS, queue backlogs, and uptime' },
    { method: 'POST', path: '/api/whatsapp/link', desc: 'Generate verified WhatsApp notification link' },
    { method: 'POST', path: '/api/emergency/call', desc: 'Dispatch Twilio voice call to emergency services' },
    { method: 'POST', path: '/api/emergency/message', desc: 'Dispatch Twilio emergency SMS alert' }
  ];

  return (
    <div className="settings-doc-page" id="page-settings-doc">
      <div className="page-header" style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)' }}>Settings & Documentation</h2>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          System architecture specifications, upload limits, privacy rules, API explorer, and troubleshooting guide.
        </p>
      </div>

      {/* Navigation Sub-Tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '20px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
        {[
          { id: 'docs', label: 'Architecture & Rules', icon: '📖' },
          { id: 'api', label: 'API Reference', icon: '🔌' },
          { id: 'privacy', label: 'Privacy & Storage Policy', icon: '🔒' },
          { id: 'troubleshooting', label: 'Troubleshooting Guide', icon: '🛠️' }
        ].map(t => (
          <button
            key={t.id}
            className={`btn ${activeTab === t.id ? 'btn-primary' : 'btn-secondary'} btn-sm`}
            onClick={() => setActiveTab(t.id)}
            id={`tab-settings-${t.id}`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* Tab 1: Architecture & Rules */}
      {activeTab === 'docs' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="glass-card">
            <div className="card-title-row">
              <div className="card-title">
                <span>📹</span>
                <span>Video Ingestion & Preprocessing Contracts</span>
              </div>
            </div>
            <div style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--text-body)' }}>
              <p>
                To maintain compatibility with the preserved <code>best_multimodal_model.keras</code> artifact:
              </p>
              <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
                <li><strong>Supported Containers:</strong> MP4, AVI, MOV, MKV, WEBM with standard H.264/H.265/VP9 codecs.</li>
                <li><strong>Maximum Upload Size:</strong> 100 MB per video.</li>
                <li><strong>Temporal Sampling Policy:</strong> Exactly 12 frames are extracted at uniform intervals $\tau_i = \frac{i}{12} \times \text{duration}$.</li>
                <li><strong>Frame Resizing & Normalization:</strong> Resized to $224 \times 224$ pixels, RGB channel order, float32 scaled to $[0.0, 1.0]$.</li>
              </ul>
            </div>
          </div>

          <div className="glass-card">
            <div className="card-title-row">
              <div className="card-title">
                <span>🎧</span>
                <span>Acoustic & Speech Recognition Rules</span>
              </div>
            </div>
            <div style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--text-body)' }}>
              <ul style={{ paddingLeft: '20px' }}>
                <li><strong>Sampling Rate:</strong> Resampled to 16,000 Hz single-channel mono.</li>
                <li><strong>Mel Spectrogram Bins:</strong> 64 mel bins, 1024 FFT window, 256 hop size, resulting in a $(64, 313, 1)$ tensor.</li>
                <li><strong>Missing Audio Policy:</strong> If a video lacks an audio track, the acoustic branch receives zero values and the <code>audio_mask</code> tensor is set to $0.0$, preventing synthetic hallucination.</li>
                <li><strong>ASR Anti-Hallucination:</strong> Near-silent audio (RMS &lt; 0.001) triggers immediate silence gating, returning an empty transcript. Repetition loops and high speech rates are dropped.</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: API Reference */}
      {activeTab === 'api' && (
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🔌</span>
              <span>REST API Endpoints Specification</span>
            </div>
            <span className="badge badge-normal">FastAPI 0.115</span>
          </div>

          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>Path</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {apiEndpoints.map((ep, i) => (
                  <tr key={i}>
                    <td>
                      <span className={`badge ${ep.method === 'GET' ? 'badge-info' : 'badge-normal'}`}>
                        {ep.method}
                      </span>
                    </td>
                    <td className="font-mono" style={{ fontSize: '12.5px', color: 'var(--text-main)' }}>
                      {ep.path}
                    </td>
                    <td style={{ fontSize: '13px', color: 'var(--text-body)' }}>
                      {ep.desc}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Privacy & Storage Policy */}
      {activeTab === 'privacy' && (
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🔒</span>
              <span>Data Protection & Privacy Architecture</span>
            </div>
          </div>

          <div style={{ fontSize: '13.5px', lineHeight: 1.8, color: 'var(--text-body)' }}>
            <p>
              The system employs strict confidentiality and path traversal security controls:
            </p>
            <ul style={{ paddingLeft: '20px', marginTop: '10px' }}>
              <li>
                <strong>No Public Filesystem Exposure:</strong> Internal operating system filepaths (e.g., <code>C:\Users\...</code>) are strictly quarantined in internal database rows and never serialized in public API responses.
              </li>
              <li>
                <strong>Canonical Artifact Resolvers:</strong> Media artifacts are accessed via relative identifiers (e.g., <code>/api/jobs/{'{id}'}/artifacts/{'{artifact_id}'}</code>) which resolve through secure base path whitelists with directory-traversal protections.
              </li>
              <li>
                <strong>Automated Retention & Storage Sweeps:</strong> Older media artifacts are purged according to configurable time-to-live windows, ensuring local disk usage does not exceed authorized quotas.
              </li>
              <li>
                <strong>Opt-in Emergency Dispatches:</strong> WhatsApp alerts and Twilio voice calls are exclusively dispatched upon explicit operator confirmation or validated high-confidence threshold matches.
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* Tab 4: Troubleshooting Guide */}
      {activeTab === 'troubleshooting' && (
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🛠️</span>
              <span>Operational Troubleshooting Checklist</span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '13px', lineHeight: 1.7 }}>
            <div style={{ background: 'var(--bg-surface)', padding: '14px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontWeight: '700', color: 'var(--text-main)', marginBottom: '4px' }}>
                Q: Why does the analysis take 15–20 seconds on CPU?
              </div>
              <div style={{ color: 'var(--text-body)' }}>
                A: The pipeline performs 12 distinct frame perturbations for evidence-based attribution sensitivity, alongside log-mel spectrogram calculation and Whisper ASR. On GPU/CUDA systems, inference runs in under 1.5 seconds.
              </div>
            </div>

            <div style={{ background: 'var(--bg-surface)', padding: '14px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontWeight: '700', color: 'var(--text-main)', marginBottom: '4px' }}>
                Q: What happens if an uploaded video has no audio or is completely silent?
              </div>
              <div style={{ color: 'var(--text-body)' }}>
                A: The audio pipeline automatically detects missing or silent audio (RMS &lt; 0.001), supplies an empty transcript, sets the <code>audio_mask</code> tensor to 0.0, and adds an explicit note in Section 2 of the scene description without crashing.
              </div>
            </div>

            <div style={{ background: 'var(--bg-surface)', padding: '14px', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontWeight: '700', color: 'var(--text-main)', marginBottom: '4px' }}>
                Q: Are the output prediction probabilities calibrated?
              </div>
              <div style={{ color: 'var(--text-body)' }}>
                A: No. The scores represent direct softmax outputs from the cross-modal dense classification layer. The system displays an explicit transparency disclaimer noting that raw softmax values should not be treated as calibrated Bayesian probabilities.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
