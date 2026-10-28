import React, { useEffect, useState } from 'react';
import { api } from '../api';
import { StatsCard } from '../components/StatsCard';

export function OverviewPage({ onNavigateToJob, onNavigateToPage }) {
  const [stats, setStats] = useState(null);
  const [health, setHealth] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchOverviewData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [statsRes, healthRes, readinessRes, jobsRes] = await Promise.all([
        api.getSystemStats().catch(() => null),
        api.getHealth().catch(() => null),
        api.getModelReadiness().catch(() => null),
        api.listJobs({ limit: 6, offset: 0 }).catch(() => ({ jobs: [] }))
      ]);

      setStats(statsRes);
      setHealth(healthRes);
      setReadiness(readinessRes);
      setRecentJobs(jobsRes.jobs || []);
    } catch (err) {
      setError(err.message || 'Failed to load project overview');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviewData();
  }, []);

  const dbMetrics = stats?.job_engine?.database_metrics || {};
  const totalJobs = dbMetrics.total_jobs ?? 0;
  const completedJobs = dbMetrics.completed_jobs ?? 0;
  const failedJobs = dbMetrics.failed_jobs ?? 0;
  const processingJobs = (dbMetrics.processing_jobs ?? 0) + (dbMetrics.queued_jobs ?? 0);

  // Compute class counts from recent jobs
  const fireCount = recentJobs.filter(j => j.prediction === 'Fire').length;
  const accidentCount = recentJobs.filter(j => j.prediction === 'Accident').length;
  const normalCount = recentJobs.filter(j => j.prediction === 'Normal').length;

  return (
    <div className="overview-page" id="page-overview">
      <div className="page-header" style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)' }}>Project Overview</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Real-time status of the multimodal audio-visual scene analysis platform.
          </p>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={fetchOverviewData} disabled={loading} id="btn-refresh-overview">
          🔄 Refresh Status
        </button>
      </div>

      {error && (
        <div className="disclaimer-banner" style={{ background: 'rgba(255, 90, 95, 0.1)', borderColor: 'rgba(255, 90, 95, 0.3)', color: '#ff6b70', marginBottom: '20px' }}>
          <span>⚠️</span> <span>{error}</span>
        </div>
      )}

      {/* KPI Cards Row */}
      <div className="stats-grid">
        <StatsCard
          title="Total Analyses"
          value={totalJobs}
          subtext="Persistent SQLite records"
          icon="📊"
          id="stat-total-jobs"
        />
        <StatsCard
          title="Completed Jobs"
          value={completedJobs}
          subtext={failedJobs > 0 ? `${failedJobs} failed records` : (totalJobs > 0 ? `${((completedJobs / totalJobs) * 100).toFixed(0)}% success rate` : 'Standby')}
          icon="✅"
          color="#10B981"
          id="stat-completed-jobs"
        />
        <StatsCard
          title="Active / In-Queue"
          value={processingJobs}
          subtext="Worker pool processing"
          icon="⏳"
          color="#3B82F6"
          id="stat-processing-jobs"
        />
        <StatsCard
          title="Model Status"
          value={readiness?.overall_status === 'ready' ? 'Ready (Loaded)' : readiness?.overall_status || 'Checking'}
          subtext="best_multimodal_model.keras"
          icon="🧠"
          color={readiness?.overall_status === 'ready' ? '#10B981' : '#F59E0B'}
          id="stat-model-status"
        />
      </div>

      {/* Main Grid: Health Status & Class Distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        {/* Component Health Card */}
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🩺</span>
              <span>Component Health & Readiness</span>
            </div>
            <span className={`badge ${health?.healthy ? 'badge-normal' : 'badge-accident'}`}>
              {health?.healthy ? 'Healthy' : 'Degraded'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {health?.checks ? (
              Object.entries(health.checks).map(([comp, st]) => {
                const isOk = st === 'ok' || st === 'loaded' || st === 'writable';
                return (
                  <div key={comp} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: 'var(--bg-surface)', borderRadius: 'var(--radius-xs)' }}>
                    <span style={{ fontSize: '13px', textTransform: 'capitalize', color: 'var(--text-body)' }}>
                      {comp.replace('_', ' ')}
                    </span>
                    <span className={`badge ${isOk ? 'badge-normal' : 'badge-fire'}`}>
                      {st}
                    </span>
                  </div>
                );
              })
            ) : (
              <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>Health metrics not available</div>
            )}
          </div>
        </div>

        {/* Class Distribution Card */}
        <div className="glass-card">
          <div className="card-title-row">
            <div className="card-title">
              <span>🎯</span>
              <span>Sample Detections Distribution</span>
            </div>
            <span className="badge badge-neutral">Recent Activity</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '6px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(255, 90, 95, 0.08)', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(255, 90, 95, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '18px' }}>🔥</span>
                <span style={{ fontWeight: '600', color: 'var(--text-main)' }}>Fire Incidents</span>
              </div>
              <span className="font-mono" style={{ fontSize: '16px', fontWeight: '700', color: '#ff6b70' }}>{fireCount}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(245, 158, 11, 0.08)', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '18px' }}>💥</span>
                <span style={{ fontWeight: '600', color: 'var(--text-main)' }}>Accidents</span>
              </div>
              <span className="font-mono" style={{ fontSize: '16px', fontWeight: '700', color: '#fbbf24' }}>{accidentCount}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(16, 185, 129, 0.08)', borderRadius: 'var(--radius-xs)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '18px' }}>✅</span>
                <span style={{ fontWeight: '600', color: 'var(--text-main)' }}>Normal Scenes</span>
              </div>
              <span className="font-mono" style={{ fontSize: '16px', fontWeight: '700', color: '#34d399' }}>{normalCount}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Detections Table */}
      <div className="glass-card">
        <div className="card-title-row">
          <div>
            <div className="card-title">
              <span>🕒</span>
              <span>Recent Scene Analyses</span>
            </div>
            <div className="card-subtitle">Latest processed videos with predictions and verification states</div>
          </div>
          <button className="btn btn-primary btn-sm" onClick={() => onNavigateToPage('studio')} id="btn-goto-studio">
            + New Video Analysis
          </button>
        </div>

        {recentJobs.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📹</div>
            <h3>No Analyses Yet</h3>
            <p>Upload a video in the Video Analysis Studio to run genuine multimodal inference.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Filename</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Status</th>
                  <th>Created At</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {recentJobs.map(job => (
                  <tr key={job.job_id}>
                    <td className="font-mono" style={{ fontSize: '12px' }}>{job.job_id}</td>
                    <td style={{ maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {job.filename}
                    </td>
                    <td>
                      {job.prediction ? (
                        <span className={`badge ${job.prediction === 'Fire' ? 'badge-fire' : job.prediction === 'Accident' ? 'badge-accident' : 'badge-normal'}`}>
                          {job.prediction}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>Pending</span>
                      )}
                    </td>
                    <td className="font-mono">
                      {job.confidence !== null && job.confidence !== undefined ? `${(job.confidence * 100).toFixed(2)}%` : '—'}
                    </td>
                    <td>
                      <span className={`badge ${job.status === 'completed' ? 'badge-normal' : job.status === 'failed' ? 'badge-fire' : 'badge-info'}`}>
                        {job.status}
                      </span>
                    </td>
                    <td style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => onNavigateToJob(job.job_id)}
                      >
                        Inspect Evidence
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
