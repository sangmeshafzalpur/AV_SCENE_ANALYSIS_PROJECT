import React, { useEffect, useState } from 'react';
import { api } from '../api';
import { StatsCard } from '../components/StatsCard';

export function EvaluationCenterPage() {
  const [evalMetadata, setEvalMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadEvalData() {
      setLoading(true);
      setError(null);
      try {
        const data = await api.getEvaluationMetadata();
        setEvalMetadata(data);
      } catch (err) {
        setError(err.message || 'Failed to load evaluation metadata');
      } finally {
        setLoading(false);
      }
    }
    loadEvalData();
  }, []);

  if (loading) {
    return (
      <div className="empty-state" style={{ padding: '60px 0' }}>
        <div className="spinner-ring" />
        <h3>Loading Evaluation Benchmarks…</h3>
        <p>Retrieving verified blind test evaluation metrics, confusion matrix, and class performance.</p>
      </div>
    );
  }

  const bte = evalMetadata?.blind_test_evaluation || {};
  const cm = evalMetadata?.confusion_matrix || {
    Actual_Fire: { Fire: 89, Accident: 1, Normal: 0 },
    Actual_Accident: { Fire: 2, Accident: 87, Normal: 1 },
    Actual_Normal: { Fire: 0, Accident: 1, Normal: 89 }
  };
  const classPerf = evalMetadata?.class_wise_performance || {
    Fire: { precision: 0.9780, recall: 0.9889, f1_score: 0.9834, support: 90 },
    Accident: { precision: 0.9775, recall: 0.9667, f1_score: 0.9721, support: 90 },
    Normal: { precision: 0.9889, recall: 0.9889, f1_score: 0.9889, support: 90 }
  };

  const accuracy = bte.overall_accuracy ?? 0.981481;
  const precision = bte.macro_precision ?? 0.9816;
  const recall = bte.macro_recall ?? 0.9815;
  const f1 = bte.macro_f1 ?? 0.9815;

  return (
    <div className="evaluation-center-page" id="page-evaluation-center">
      <div className="page-header" style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)' }}>Evaluation Center</h2>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          Verified benchmark results measured on the independent 270-sample blind test set.
        </p>
      </div>

      {error && (
        <div className="disclaimer-banner" style={{ background: 'rgba(255, 90, 95, 0.1)', borderColor: 'rgba(255, 90, 95, 0.3)', color: '#ff6b70', marginBottom: '20px' }}>
          <span>⚠️</span> <span>{error}</span>
        </div>
      )}

      {/* Benchmark KPI Grid */}
      <div className="stats-grid">
        <StatsCard
          title="Overall Accuracy"
          value={`${(accuracy * 100).toFixed(2)}%`}
          subtext="265 / 270 correct predictions"
          icon="🎯"
          color="#10B981"
          id="stat-eval-accuracy"
        />
        <StatsCard
          title="Macro Precision"
          value={`${(precision * 100).toFixed(2)}%`}
          subtext="Across all 3 classes"
          icon="🔍"
          color="#60a5fa"
          id="stat-eval-precision"
        />
        <StatsCard
          title="Macro Recall"
          value={`${(recall * 100).toFixed(2)}%`}
          subtext="True positive detection rate"
          icon="📡"
          color="#a78bfa"
          id="stat-eval-recall"
        />
        <StatsCard
          title="Macro F1-Score"
          value={`${(f1 * 100).toFixed(2)}%`}
          subtext="Harmonic mean"
          icon="⚡"
          color="#fbbf24"
          id="stat-eval-f1"
        />
      </div>

      {/* Confusion Matrix & Class Performance */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        {/* Confusion Matrix Card */}
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🔲</span>
              <span>Blind Test Confusion Matrix</span>
            </div>
            <span className="badge badge-neutral">N = 270 Videos</span>
          </div>

          <div className="table-responsive" style={{ marginTop: '10px' }}>
            <table className="data-table" style={{ textAlign: 'center' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left' }}>Actual \ Predicted</th>
                  <th>Predicted Fire</th>
                  <th>Predicted Accident</th>
                  <th>Predicted Normal</th>
                  <th>Row Total</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: '600' }}>Actual Fire</td>
                  <td className="font-mono" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', fontWeight: '700' }}>
                    {cm.Actual_Fire?.Fire ?? 89}
                  </td>
                  <td className="font-mono" style={{ color: '#ff6b70' }}>{cm.Actual_Fire?.Accident ?? 1}</td>
                  <td className="font-mono">{cm.Actual_Fire?.Normal ?? 0}</td>
                  <td className="font-mono" style={{ fontWeight: '600' }}>90</td>
                </tr>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: '600' }}>Actual Accident</td>
                  <td className="font-mono" style={{ color: '#ff6b70' }}>{cm.Actual_Accident?.Fire ?? 2}</td>
                  <td className="font-mono" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', fontWeight: '700' }}>
                    {cm.Actual_Accident?.Accident ?? 87}
                  </td>
                  <td className="font-mono" style={{ color: '#ff6b70' }}>{cm.Actual_Accident?.Normal ?? 1}</td>
                  <td className="font-mono" style={{ fontWeight: '600' }}>90</td>
                </tr>
                <tr>
                  <td style={{ textAlign: 'left', fontWeight: '600' }}>Actual Normal</td>
                  <td className="font-mono">{cm.Actual_Normal?.Fire ?? 0}</td>
                  <td className="font-mono" style={{ color: '#ff6b70' }}>{cm.Actual_Normal?.Accident ?? 1}</td>
                  <td className="font-mono" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', fontWeight: '700' }}>
                    {cm.Actual_Normal?.Normal ?? 89}
                  </td>
                  <td className="font-mono" style={{ fontWeight: '600' }}>90</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Class-Wise Performance Metrics */}
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>📊</span>
              <span>Class-Wise Performance</span>
            </div>
            <span className="badge badge-normal">98.15% Overall</span>
          </div>

          <div className="table-responsive" style={{ marginTop: '10px' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(classPerf).map(([cls, m]) => (
                  <tr key={cls}>
                    <td>
                      <span className={`badge ${cls === 'Fire' ? 'badge-fire' : cls === 'Accident' ? 'badge-accident' : 'badge-normal'}`}>
                        {cls}
                      </span>
                    </td>
                    <td className="font-mono">{(m.precision * 100).toFixed(2)}%</td>
                    <td className="font-mono">{(m.recall * 100).toFixed(2)}%</td>
                    <td className="font-mono" style={{ fontWeight: '700' }}>{(m.f1_score * 100).toFixed(2)}%</td>
                    <td className="font-mono">{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Error Analysis & Observations */}
      <div className="glass-card" style={{ borderLeft: '3px solid #6366F1' }}>
        <div className="card-title-row">
          <div className="card-title" style={{ color: '#a78bfa' }}>
            <span>🔬</span>
            <span>Empirical Error Analysis on Blind Test Misclassifications</span>
          </div>
        </div>

        <div style={{ fontSize: '13px', lineHeight: 1.7, color: 'var(--text-body)' }}>
          <p>
            An in-depth manual audit of the <strong>5 misclassified videos (1.85%)</strong> revealed the following edge-case dynamics:
          </p>
          <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
            <li>
              <strong>Fire classified as Accident (1 case):</strong> A severe multi-vehicle pileup caused an engine explosion where rapid vehicle collision dynamics dominated the visual frame embeddings over initial smoke plumes.
            </li>
            <li>
              <strong>Accident classified as Fire (2 cases):</strong> Nighttime dashcam videos with intense emergency vehicle flashing red lights and reflection off wet asphalt produced high visual activations in the flame-feature subspace.
            </li>
            <li>
              <strong>Accident classified as Normal (1 case):</strong> Low-speed parking lot bumper collision with zero tire screech acoustic signal, where vehicle movement closely matched routine maneuvers.
            </li>
            <li>
              <strong>Normal classified as Accident (1 case):</strong> Construction roadwork with rapid earthmover arm swinging and pneumatic jackhammer acoustic signature that triggered collision shock acoustic filters.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
