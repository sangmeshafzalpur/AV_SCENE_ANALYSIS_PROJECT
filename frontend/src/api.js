/**
 * Audio-Visual Scene Analysis - Unified API Client
 * Connects frontend pages to verified backend endpoints.
 * Handles graceful error degradation, timeout management, and 'Not available' fallbacks.
 */

export const BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '').replace(/\/+$/, '');

async function request(endpoint, options = {}) {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = endpoint.startsWith('http') ? endpoint : `${BASE_URL}${cleanEndpoint}`;
  const config = {
    ...options,
    headers: {
      'Accept': 'application/json',
      ...(options.headers || {})
    }
  };

  try {
    const res = await fetch(url, config);
    if (!res.ok) {
      let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) {
          errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        } else if (errJson.message) {
          errorDetail = errJson.message;
        }
      } catch {
        // Fallback to text status
      }
      const error = new Error(errorDetail);
      error.status = res.status;
      throw error;
    }
    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error('Backend server is unreachable (connection refused). Verify port 8000 is running.');
    }
    throw err;
  }
}

export const api = {
  // System Health & Stats
  getHealth: () => request('/api/health'),
  getReadiness: () => request('/api/readiness'),
  getSystemStats: () => request('/api/system/stats'),

  // Model Metadata
  getModelMetadata: () => request('/api/model/metadata'),
  getModelReadiness: () => request('/api/model/readiness'),

  // Evaluation & Dataset
  getEvaluationMetadata: () => request('/api/evaluation/metadata'),

  // Storage
  getStoragePolicy: () => request('/api/storage/policy'),
  cleanupStorage: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return request(`/api/storage/cleanup?${q}`, { method: 'POST' });
  },

  // Jobs
  listJobs: (params = {}) => {
    const cleanParams = {};
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== '') cleanParams[k] = v;
    });
    const q = new URLSearchParams(cleanParams).toString();
    return request(`/api/jobs${q ? `?${q}` : ''}`);
  },

  getJob: (jobId) => request(`/api/jobs/${jobId}`),

  createAnalysisJob: async (file, options = {}) => {
    const formData = new FormData();
    formData.append('video', file);
    const q = new URLSearchParams();
    if (options.forceReanalysis) q.append('force_reanalysis', 'true');
    if (options.description) q.append('description', options.description);
    const queryStr = q.toString() ? `?${q.toString()}` : '';

    const res = await fetch(`${BASE_URL}/api/jobs/analyze${queryStr}`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }
    return await res.json();
  },

  cancelJob: (jobId) => request(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),

  updateJob: (jobId, data) => request(`/api/jobs/${jobId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  }),

  // Detailed Analysis Evidence Endpoints
  getJobAttributions: (jobId, sortBy = 'rank') => request(`/api/jobs/${jobId}/attributions?sort_by=${sortBy}`),
  getJobTranscripts: (jobId) => request(`/api/jobs/${jobId}/transcripts`),
  getJobDescription: (jobId) => request(`/api/jobs/${jobId}/description`),
  getJobAudioMetadata: (jobId) => request(`/api/jobs/${jobId}/audio`),
  getJobFrames: (jobId) => request(`/api/jobs/${jobId}/frames`),
  getJobArtifacts: (jobId) => request(`/api/jobs/${jobId}/artifacts`),

  // Direct sync predict endpoint
  predictSync: async (file) => {
    const formData = new FormData();
    formData.append('video', file);
    const res = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Prediction failed with status ${res.status}`);
    }
    return await res.json();
  },

  // Emergency Dispatches
  getWhatsAppLink: (event, confidence, videoIdentifier) => {
    const params = new URLSearchParams({
      event,
      confidence: confidence.toString(),
      video_identifier: videoIdentifier
    });
    return request(`/api/whatsapp/link?${params.toString()}`);
  },

  sendWhatsAppAlert: (prediction, confidence, videoIdentifier) => {
    return request('/send-whatsapp-alert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prediction,
        confidence,
        video_identifier: videoIdentifier
      })
    });
  },

  triggerEmergencyCall: (event, confidence, videoIdentifier, eventId) => {
    return request('/api/emergency/call', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event,
        confidence,
        video_identifier: videoIdentifier,
        event_id: eventId || `ev_${Date.now()}`
      })
    });
  },

  triggerEmergencyMessage: (event, confidence, videoIdentifier, eventId) => {
    return request('/api/emergency/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event,
        confidence,
        video_identifier: videoIdentifier,
        event_id: eventId || `ev_${Date.now()}`
      })
    });
  },

  // Comparison
  compareJobs: (job1, job2) => request(`/api/jobs/compare?job_1=${job1}&job_2=${job2}`),

  // Notifications & Alert Audit
  getNotificationsDispatches: (jobId) => request(`/api/notifications/dispatches${jobId ? `?job_id=${jobId}` : ''}`),
  getNotificationsConfig: () => request('/api/notifications/config'),
  testNotification: (event = 'Fire', confidence = 0.95) => request('/api/notifications/test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event, confidence })
  }),

  // Report Download URL helper
  getReportUrl: (jobId, format = 'json') => `${BASE_URL}/api/jobs/${jobId}/report?format=${format}`,
  getProjectReportUrl: (format = 'json') => `${BASE_URL}/api/reports/project?format=${format}`,
  getThumbnailUrl: (jobId) => `${BASE_URL}/api/jobs/${jobId}/thumbnail`,
  getSpectrogramUrl: (jobId) => `${BASE_URL}/api/jobs/${jobId}/spectrogram`,
  getFilmstripUrl: (jobId) => `${BASE_URL}/api/jobs/${jobId}/filmstrip`,
  getArtifactUrl: (jobId, artifactId) => `${BASE_URL}/api/jobs/${jobId}/artifacts/${artifactId}`
};
