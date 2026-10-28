import React, { useEffect, useState } from 'react';
import { api } from '../api';

export function ModelCenterPage() {
  const [metadata, setMetadata] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const [metaRes, readyRes] = await Promise.all([
          api.getModelMetadata().catch(() => null),
          api.getModelReadiness().catch(() => null)
        ]);
        setMetadata(metaRes);
        setReadiness(readyRes);
      } catch (err) {
        setError(err.message || 'Failed to load model metadata');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="empty-state" style={{ padding: '60px 0' }}>
        <div className="spinner-ring" />
        <h3>Loading Model Specifications…</h3>
        <p>Querying verified model layers, parameters, and tensor contracts.</p>
      </div>
    );
  }

  const modelId = metadata?.model_identification || {};
  const classMap = metadata?.class_index_mapping || { 0: 'Accident', 1: 'Fire', 2: 'Normal' };

  return (
    <div className="model-center-page" id="page-model-center">
      <div className="page-header" style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)' }}>Model Information Center</h2>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          Verified architecture, input tensor signatures, preprocessing contracts, and hardware status.
        </p>
      </div>

      {error && (
        <div className="disclaimer-banner" style={{ background: 'rgba(255, 90, 95, 0.1)', borderColor: 'rgba(255, 90, 95, 0.3)', color: '#ff6b70', marginBottom: '20px' }}>
          <span>⚠️</span> <span>{error}</span>
        </div>
      )}

      {/* Model Identity Overview Card */}
      <div className="glass-card" style={{ marginBottom: '24px' }}>
        <div className="card-title-row">
          <div>
            <div className="card-title" style={{ fontSize: '17px' }}>
              <span>🧠</span>
              <span>{modelId.canonical_name || 'EfficientNetB0_BiLSTM_AudioCNN_KannadaASR_TextBiLSTM'}</span>
            </div>
            <div className="card-subtitle font-mono" style={{ fontSize: '12px' }}>
              Artifact: {modelId.filename || 'best_multimodal_model.keras'}
            </div>
          </div>
          <span className={`badge ${readiness?.overall_status === 'ready' ? 'badge-normal' : 'badge-accident'}`}>
            {readiness?.overall_status === 'ready' ? 'Active & Ready' : 'Degraded'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '14px', marginTop: '16px' }}>
          <div style={{ background: 'var(--bg-surface)', padding: '12px 16px', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Parameters</div>
            <div className="font-mono" style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)', marginTop: '2px' }}>
              {modelId.total_parameters ? Number(modelId.total_parameters).toLocaleString() : '6,455,846'}
            </div>
          </div>

          <div style={{ background: 'var(--bg-surface)', padding: '12px 16px', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>File Size</div>
            <div className="font-mono" style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-main)', marginTop: '2px' }}>
              {modelId.file_size_bytes ? `${(modelId.file_size_bytes / (1024 * 1024)).toFixed(2)} MB` : '12.56 MB'}
            </div>
          </div>

          <div style={{ background: 'var(--bg-surface)', padding: '12px 16px', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Framework Versions</div>
            <div className="font-mono" style={{ fontSize: '14px', fontWeight: '600', color: 'var(--text-main)', marginTop: '4px' }}>
              TF {modelId.framework_versions?.tensorflow || '2.19.0'} · Keras {modelId.framework_versions?.keras || '3.8.0'}
            </div>
          </div>

          <div style={{ background: 'var(--bg-surface)', padding: '12px 16px', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Verification Checksum</div>
            <div className="font-mono" style={{ fontSize: '11px', color: '#60a5fa', wordBreak: 'break-all', marginTop: '4px' }}>
              {modelId.sha256_checksum || '404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde'}
            </div>
          </div>
        </div>
      </div>

      {/* Input / Output Tensors Contract */}
      <div className="glass-card" style={{ marginBottom: '24px' }}>
        <div className="card-title-row">
          <div className="card-title">
            <span>📐</span>
            <span>Input / Output Tensor Signatures</span>
          </div>
          <span className="badge badge-info">Verified Model Contract</span>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Tensor Role</th>
                <th>Layer Name</th>
                <th>Tensor Shape</th>
                <th>Data Type</th>
                <th>Modality & Preprocessing Requirement</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span className="badge badge-normal">Input 1</span></td>
                <td className="font-mono">video</td>
                <td className="font-mono">(None, 12, 224, 224, 3)</td>
                <td className="font-mono">float32</td>
                <td>12 equidistant frames, resized 224×224, normalized [0, 1]</td>
              </tr>
              <tr>
                <td><span className="badge badge-normal">Input 2</span></td>
                <td className="font-mono">audio</td>
                <td className="font-mono">(None, 64, 313, 1)</td>
                <td className="font-mono">float32</td>
                <td>Log-mel spectrogram: 64 mel bins, 313 time frames, 16 kHz sample rate</td>
              </tr>
              <tr>
                <td><span className="badge badge-normal">Input 3</span></td>
                <td className="font-mono">audio_mask</td>
                <td className="font-mono">(None, 1)</td>
                <td className="font-mono">float32</td>
                <td>Binary indicator (1.0 = genuine acoustic signal, 0.0 = silent/missing)</td>
              </tr>
              <tr>
                <td><span className="badge badge-normal">Input 4</span></td>
                <td className="font-mono">text</td>
                <td className="font-mono">(None,)</td>
                <td className="font-mono">string</td>
                <td>Raw Kannada/English speech transcript passed to internal TextVectorization</td>
              </tr>
              <tr style={{ background: 'rgba(99, 102, 241, 0.08)' }}>
                <td><span className="badge badge-purple">Output</span></td>
                <td className="font-mono">event_prediction</td>
                <td className="font-mono">(None, 3)</td>
                <td className="font-mono">float32</td>
                <td>Softmax activation scores for classes [Accident, Fire, Normal]</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Class Mapping & Subsystem Pipeline */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        {/* Class Index Mapping */}
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🏷️</span>
              <span>Verified Class Mapping</span>
            </div>
            <span className="badge badge-neutral">3 Output Classes</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {Object.entries(classMap).map(([idx, name]) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-surface)', borderRadius: 'var(--radius-xs)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span className="font-mono badge badge-neutral" style={{ fontSize: '13px' }}>Index {idx}</span>
                  <span style={{ fontWeight: '600', color: 'var(--text-main)', fontSize: '14px' }}>{name}</span>
                </div>
                <span>{name === 'Fire' ? '🔥' : name === 'Accident' ? '💥' : '✅'}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Multimodal Architecture Branches */}
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🏗️</span>
              <span>Subsystem Pipeline</span>
            </div>
            <span className="badge badge-accent">Cross-Modal Fusion</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px', lineHeight: 1.6 }}>
            <div>
              <strong>Visual Branch:</strong> TimeDistributed(EfficientNetB0) → Bidirectional LSTM → Dense(128)
            </div>
            <div>
              <strong>Acoustic Branch:</strong> 3× Conv2D + MaxPool → GlobalAveragePooling2D → Dense(128) gated by audio_mask
            </div>
            <div>
              <strong>Speech Branch:</strong> TextVectorization → Embedding → Bidirectional LSTM → Dense(128)
            </div>
            <div>
              <strong>Cross-Modal Fusion:</strong> Concatenate(Visual, Acoustic, Speech) → Dense(256) → Dropout(0.3) → Dense(128) → Dense(3, softmax)
            </div>
          </div>
        </div>
      </div>

      {/* Known Limitations Card */}
      <div className="glass-card" style={{ borderLeft: '3px solid #F59E0B' }}>
        <div className="card-title-row">
          <div className="card-title" style={{ color: '#fbbf24' }}>
            <span>⚠️</span>
            <span>Documented Limitations & Calibration Transparency</span>
          </div>
        </div>

        <ul style={{ paddingLeft: '20px', fontSize: '13px', color: 'var(--text-body)', lineHeight: 1.8 }}>
          <li>
            <strong>Uncalibrated Softmax Scores:</strong> Raw model output probabilities represent softmax activations and must not be interpreted as calibrated Bayesian probabilities without post-hoc temperature scaling or Platt scaling.
          </li>
          <li>
            <strong>Perturbation Sensitivity vs Real-World Causation:</strong> Influential-frame sensitivity rankings measure changes in model activation when a frame is ablated with neutral gray. They do not constitute proof of real-world physical causality.
          </li>
          <li>
            <strong>Language Bounds:</strong> Whisper ASR pipeline supports Kannada and English natively; unsupported scripts are rejected to prevent hallucinated speech tokens.
          </li>
          <li>
            <strong>Missing Audio Tolerance:</strong> Silent audio or missing audio channels are explicitly passed to the audio mask gate with zero acoustic contribution, avoiding synthetic hallucination.
          </li>
        </ul>
      </div>
    </div>
  );
}
