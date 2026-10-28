import React, { useEffect, useState } from 'react';
import { api } from '../api';

export function HistoryReportsPage({ onInspectJob }) {
  const [jobs, setJobs] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters & Search
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [predFilter, setPredFilter] = useState('');
  const [limit, setLimit] = useState(15);
  const [offset, setOffset] = useState(0);

  // Comparison State
  const [compareIds, setCompareIds] = useState([]);
  const [compareData, setCompareData] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const fetchJobs = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listJobs({
        query: query.trim() || undefined,
        status: statusFilter || undefined,
        prediction: predFilter || undefined,
        limit,
        offset
      });
      setJobs(res.jobs || []);
      setTotal(res.total || 0);
    } catch (err) {
      setError(err.message || 'Failed to fetch job history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter, predFilter, offset, limit]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setOffset(0);
    fetchJobs();
  };

  const handleToggleCompare = (jobId) => {
    if (compareIds.includes(jobId)) {
      setCompareIds(compareIds.filter(id => id !== jobId));
    } else {
      if (compareIds.length >= 2) {
        setCompareIds([compareIds[1], jobId]);
      } else {
        setCompareIds([...compareIds, jobId]);
      }
    }
  };

  const handleOpenComparison = async () => {
    if (compareIds.length !== 2) return;
    setCompareLoading(true);
    try {
      const cmp = await api.compareJobs(compareIds[0], compareIds[1]);
      setCompareData(cmp);
    } catch (err) {
      alert(`Failed to load comparison: ${err.message}`);
    } finally {
      setCompareLoading(false);
    }
  };

  const formatProb = (val) => (val !== undefined && val !== null ? `${(val * 100).toFixed(2)}%` : '—');

  return (
    <div className="history-reports-page" id="page-history-reports">
      <div className="page-header" style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <h2 style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)' }}>History and Reports</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Searchable persistent analysis records, multi-format report exports, and side-by-side job comparison.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'var(--bg-surface)', padding: '6px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-muted)' }}>📊 Project Report:</span>
            <a href={api.getProjectReportUrl('pdf')} download="project_level_report.pdf" className="btn btn-secondary btn-sm" style={{ padding: '3px 8px', fontSize: '11px' }} id="btn-project-report-pdf">PDF</a>
            <a href={api.getProjectReportUrl('csv')} download="project_level_report.csv" className="btn btn-secondary btn-sm" style={{ padding: '3px 8px', fontSize: '11px' }} id="btn-project-report-csv">CSV</a>
            <a href={api.getProjectReportUrl('html')} download="project_level_report.html" className="btn btn-secondary btn-sm" style={{ padding: '3px 8px', fontSize: '11px' }} id="btn-project-report-html">HTML</a>
            <a href={api.getProjectReportUrl('json')} download="project_level_report.json" className="btn btn-secondary btn-sm" style={{ padding: '3px 8px', fontSize: '11px' }} id="btn-project-report-json">JSON</a>
          </div>

          {compareIds.length === 2 && (
            <button
              className="btn btn-accent btn-sm"
              onClick={handleOpenComparison}
              disabled={compareLoading}
              id="btn-compare-jobs"
            >
              ⚖️ Compare 2 Selected Jobs
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="disclaimer-banner" style={{ background: 'rgba(255, 90, 95, 0.1)', borderColor: 'rgba(255, 90, 95, 0.3)', color: '#ff6b70', marginBottom: '20px' }}>
          <span>⚠️</span> <span>{error}</span>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="glass-card" style={{ marginBottom: '20px', padding: '16px 20px' }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center' }}>
          <div style={{ flex: '1 1 240px' }}>
            <input
              type="text"
              className="form-control"
              placeholder="Search by filename or job ID..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              id="input-search-jobs"
            />
          </div>

          <div style={{ width: '160px' }}>
            <select
              className="form-control"
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value); setOffset(0); }}
              id="select-filter-status"
            >
              <option value="">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="processing">Processing</option>
              <option value="queued">Queued</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>

          <div style={{ width: '160px' }}>
            <select
              className="form-control"
              value={predFilter}
              onChange={(e) => { setPredFilter(e.target.value); setOffset(0); }}
              id="select-filter-pred"
            >
              <option value="">All Classes</option>
              <option value="Fire">Fire</option>
              <option value="Accident">Accident</option>
              <option value="Normal">Normal</option>
            </select>
          </div>

          <div style={{ width: '130px' }}>
            <select
              className="form-control"
              value={limit}
              onChange={(e) => { setLimit(Number(e.target.value)); setOffset(0); }}
              id="select-filter-limit"
            >
              <option value="10">10 / page</option>
              <option value="15">15 / page</option>
              <option value="25">25 / page</option>
              <option value="50">50 / page</option>
            </select>
          </div>

          <button type="submit" className="btn btn-primary btn-sm" id="btn-submit-search">
            🔍 Search
          </button>

          {(query || statusFilter || predFilter) && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setQuery('');
                setStatusFilter('');
                setPredFilter('');
                setOffset(0);
              }}
            >
              Reset
            </button>
          )}
        </form>
      </div>

      {/* Comparison Modal Drawer if active */}
      {compareData && (
        <div className="modal-overlay" onClick={() => setCompareData(null)}>
          <div className="modal-content" style={{ maxWidth: '960px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-main)' }}>
                ⚖️ Side-by-Side Job Comparison ({compareData.comparison_id || 'COMP'})
              </h3>
              <button className="btn btn-secondary btn-sm" onClick={() => setCompareData(null)}>✕</button>
            </div>
            <div className="modal-body">
              {compareData.deltas && (
                <div style={{ background: 'var(--bg-surface)', padding: '12px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', marginBottom: '18px', display: 'flex', flexWrap: 'wrap', gap: '16px', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Class Match: </span>
                    <strong style={{ color: compareData.deltas.prediction_match ? '#10B981' : '#f87171' }}>
                      {compareData.deltas.prediction_match ? '✅ Identical Prediction' : '⚠️ Divergent Predictions'}
                    </strong>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Confidence &Delta;: </span>
                    <strong className="font-mono">{compareData.deltas.confidence_delta > 0 ? `+${compareData.deltas.confidence_delta}` : compareData.deltas.confidence_delta}</strong>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Fire &Delta;: </span>
                    <span className="font-mono">{compareData.deltas.fire_score_delta}</span>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Accident &Delta;: </span>
                    <span className="font-mono">{compareData.deltas.accident_score_delta}</span>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Normal &Delta;: </span>
                    <span className="font-mono">{compareData.deltas.normal_score_delta}</span>
                  </div>
                </div>
              )}

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {[compareData.job_1 || compareData.job1, compareData.job_2 || compareData.job2].map((j, i) => (
                  <div key={j.job_id || i} style={{ background: 'var(--bg-surface)', padding: '16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Candidate #{i + 1}</div>
                    <h4 style={{ fontSize: '16px', color: 'var(--text-main)', marginTop: '4px', wordBreak: 'break-all' }}>{j.filename}</h4>
                    <div className="font-mono" style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '12px' }}>{j.job_id}</div>

                    <div style={{ marginBottom: '12px' }}>
                      <span className={`badge ${j.prediction === 'Fire' ? 'badge-fire' : j.prediction === 'Accident' ? 'badge-accident' : 'badge-normal'}`}>
                        {j.prediction || 'Pending'} ({formatProb(j.confidence)})
                      </span>
                    </div>

                    <div style={{ fontSize: '12.5px', lineHeight: 1.7, marginBottom: '14px' }}>
                      <div>• Fire Score: <span className="font-mono">{formatProb(j.probabilities?.Fire ?? j.prob_fire)}</span></div>
                      <div>• Accident Score: <span className="font-mono">{formatProb(j.probabilities?.Accident ?? j.prob_accident)}</span></div>
                      <div>• Normal Score: <span className="font-mono">{formatProb(j.probabilities?.Normal ?? j.prob_normal)}</span></div>
                      <div>• Audio Duration: <span className="font-mono">{j.audio_duration_sec ? `${j.audio_duration_sec}s` : 'N/A'}</span></div>
                    </div>

                    {j.transcript && (
                      <div style={{ background: 'var(--bg-base)', padding: '10px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)', marginBottom: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '2px' }}>Kannada/English Transcript:</div>
                        <div style={{ fontSize: '12px', color: '#38bdf8' }}>{j.transcript}</div>
                      </div>
                    )}

                    {j.top_influential_frames && j.top_influential_frames.length > 0 && (
                      <div style={{ background: 'var(--bg-base)', padding: '10px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Top Influential Frames:</div>
                        <div style={{ fontSize: '11.5px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          {j.top_influential_frames.map((tf, tfi) => (
                            <div key={tfi} style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span>Frame #{tf.frame_index} ({tf.timestamp_sec?.toFixed(1)}s)</span>
                              <span className="font-mono" style={{ color: '#fbbf24' }}>Rank #{tf.rank}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Jobs Records Table */}
      <div className="glass-card">
        <div className="card-title-row">
          <div>
            <div className="card-title">
              <span>🗃️</span>
              <span>Analysis Records ({total} Total)</span>
            </div>
            <div className="card-subtitle">
              Showing {jobs.length > 0 ? offset + 1 : 0}–{Math.min(offset + jobs.length, total)} of {total} records
            </div>
          </div>
          {compareIds.length > 0 && (
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              {compareIds.length}/2 selected for comparison
            </div>
          )}
        </div>

        {loading ? (
          <div className="empty-state">
            <div className="spinner-ring" />
            <p>Loading records…</p>
          </div>
        ) : jobs.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📂</div>
            <h3>No Records Match Filter</h3>
            <p>Try modifying your search query or reset the filter dropdowns.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '40px' }}>Compare</th>
                  <th>Job ID</th>
                  <th>Filename</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Audio / Speech</th>
                  <th>Created</th>
                  <th>Reports</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map(job => {
                  const isChecked = compareIds.includes(job.job_id);
                  return (
                    <tr key={job.job_id}>
                      <td style={{ textAlign: 'center' }}>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => handleToggleCompare(job.job_id)}
                          aria-label={`Select job ${job.job_id} for comparison`}
                        />
                      </td>
                      <td className="font-mono" style={{ fontSize: '12px' }}>{job.job_id}</td>
                      <td style={{ maxWidth: '170px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={job.filename}>
                        {job.filename}
                      </td>
                      <td>
                        {job.prediction ? (
                          <span className={`badge ${job.prediction === 'Fire' ? 'badge-fire' : job.prediction === 'Accident' ? 'badge-accident' : 'badge-normal'}`}>
                            {job.prediction}
                          </span>
                        ) : (
                          <span className={`badge ${job.status === 'failed' ? 'badge-fire' : 'badge-info'}`}>
                            {job.status}
                          </span>
                        )}
                      </td>
                      <td className="font-mono">{formatProb(job.confidence)}</td>
                      <td>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          {job.audio_modality_available ? '🎵' : '🔇'} · {job.speech_modality_available ? '🗣️' : '—'}
                        </span>
                      </td>
                      <td style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                        {job.created_at ? new Date(job.created_at).toLocaleDateString() : '—'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '4px' }}>
                          <a
                            href={api.getReportUrl(job.job_id, 'pdf')}
                            download={`report_${job.job_id}.pdf`}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 6px', fontSize: '10px' }}
                            title="Download Printable PDF Forensic Report"
                          >
                            PDF
                          </a>
                          <a
                            href={api.getReportUrl(job.job_id, 'csv')}
                            download={`report_${job.job_id}.csv`}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 6px', fontSize: '10px' }}
                            title="Download CSV Forensic Report (Kannada UTF-8 BOM)"
                          >
                            CSV
                          </a>
                          <a
                            href={api.getReportUrl(job.job_id, 'html')}
                            download={`report_${job.job_id}.html`}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 6px', fontSize: '10px' }}
                            title="Download Standalone HTML Report"
                          >
                            HTML
                          </a>
                          <a
                            href={api.getReportUrl(job.job_id, 'json')}
                            download={`report_${job.job_id}.json`}
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '3px 6px', fontSize: '10px' }}
                            title="Download Full JSON Artifact"
                          >
                            JSON
                          </a>
                        </div>
                      </td>
                      <td>
                        <button
                          className="btn btn-primary btn-sm"
                          style={{ padding: '4px 10px', fontSize: '12px' }}
                          onClick={() => onInspectJob(job.job_id)}
                        >
                          Studio
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Bar */}
        {total > limit && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '16px', paddingTop: '14px', borderTop: '1px solid var(--border-subtle)' }}>
            <button
              className="btn btn-secondary btn-sm"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - limit))}
              id="btn-pagination-prev"
            >
              ← Previous Page
            </button>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Page {Math.floor(offset / limit) + 1} of {Math.ceil(total / limit)}
            </div>
            <button
              className="btn btn-secondary btn-sm"
              disabled={offset + limit >= total}
              onClick={() => setOffset(offset + limit)}
              id="btn-pagination-next"
            >
              Next Page →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
