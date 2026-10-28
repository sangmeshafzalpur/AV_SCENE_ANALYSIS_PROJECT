import React, { useState, useRef, useEffect } from 'react';
import { api } from '../api';
import { ProbabilityBar } from '../components/ProbabilityBar';
import { FrameViewer } from '../components/FrameViewer';
import { AttributionViewer } from '../components/AttributionViewer';
import { AudioSpectrogramViewer } from '../components/AudioSpectrogramViewer';
import { TranscriptViewer } from '../components/TranscriptViewer';
import { SceneDescriptionViewer } from '../components/SceneDescriptionViewer';
import { EmergencyActions } from '../components/EmergencyActions';
import { DetectionPopup } from '../components/DetectionPopup';

export function AnalysisStudioPage({ initialJobId, onClearInitialJob }) {
  const [file, setFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Active Job Results
  const [activeJob, setActiveJob] = useState(null);
  const [frames, setFrames] = useState([]);
  const [attributionsData, setAttributionsData] = useState(null);
  const [audioData, setAudioData] = useState(null);
  const [transcripts, setTranscripts] = useState([]);
  const [descriptionData, setDescriptionData] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'frames' | 'attribution' | 'audio' | 'speech' | 'description'

  // Detection Alert Popup State (Strictly for Fire and Accident)
  const [detectionPopup, setDetectionPopup] = useState({
    isOpen: false,
    prediction: null,
    confidence: null,
    filename: null,
    jobId: null
  });
  const [dismissedPopupJobId, setDismissedPopupJobId] = useState(null);

  const fileInputRef = useRef(null);

  const loadExistingJob = async (jobId) => {
    setLoading(true);
    setError(null);
    try {
      const [jobRes, framesRes, attrRes, audioRes, transRes, descRes] = await Promise.all([
        api.getJob(jobId),
        api.getJobFrames(jobId).catch(() => ({ frames: [] })),
        api.getJobAttributions(jobId).catch(() => null),
        api.getJobAudioMetadata(jobId).catch(() => null),
        api.getJobTranscripts(jobId).catch(() => ({ transcripts: [] })),
        api.getJobDescription(jobId).catch(() => null)
      ]);

      setActiveJob(jobRes);
      setFrames(framesRes.frames || []);
      setAttributionsData(attrRes);
      setAudioData(audioRes);
      setTranscripts(transRes.transcripts || []);
      setDescriptionData(descRes);
    } catch (err) {
      setError(err.message || `Failed to load job ${jobId}`);
    } finally {
      setLoading(false);
    }
  };

  // Load initialJobId if passed
  useEffect(() => {
    if (initialJobId) {
      loadExistingJob(initialJobId);
    }
  }, [initialJobId]);

  const validateFile = (f) => {
    if (!f) return false;
    const validExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
    if (!validExts.includes(ext)) {
      setError('Unsupported video format. Please upload MP4, AVI, MOV, MKV, or WEBM.');
      return false;
    }
    if (f.size > 100 * 1024 * 1024) {
      setError('File size exceeds the 100 MB limit.');
      return false;
    }
    setError(null);
    return true;
  };

  const handleFileSelect = (selectedFile) => {
    if (validateFile(selectedFile)) {
      setFile(selectedFile);
      setVideoPreviewUrl(URL.createObjectURL(selectedFile));
      setActiveJob(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const resetState = () => {
    setFile(null);
    if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
    setVideoPreviewUrl(null);
    setActiveJob(null);
    setFrames([]);
    setAttributionsData(null);
    setAudioData(null);
    setTranscripts([]);
    setDescriptionData(null);
    setError(null);
    setLoading(false);
    setProcessingStatus('');
    setDismissedPopupJobId(null);
    setDetectionPopup({ isOpen: false, prediction: null, confidence: null, filename: null, jobId: null });
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (onClearInitialJob) onClearInitialJob();
  };

  const handleStartAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setProcessingStatus('Uploading video and initializing asynchronous analysis engine…');

    try {
      // 1. Submit asynchronous job
      const submitRes = await api.createAnalysisJob(file, { forceReanalysis: true });
      const jobId = submitRes.job_id;

      // 2. Poll job status until completed
      const maxPoll = 90; // seconds
      let completed = false;
      let finalJob = null;
      const startTime = Date.now();

      while ((Date.now() - startTime) / 1000 < maxPoll) {
        const pollJob = await api.getJob(jobId);
        finalJob = pollJob;
        setUploadProgress(Math.min(Math.round(pollJob.progress * 100), 99));
        setProcessingStatus(pollJob.current_step || 'Processing video analysis pipeline…');

        if (pollJob.status === 'completed') {
          completed = true;
          setUploadProgress(100);
          break;
        } else if (pollJob.status === 'failed') {
          throw new Error(pollJob.error_message || 'Video analysis pipeline encountered an error.');
        } else if (pollJob.status === 'cancelled') {
          throw new Error('Analysis job was cancelled.');
        }

        await new Promise(r => setTimeout(r, 1500));
      }

      if (!completed && finalJob?.status !== 'completed') {
        throw new Error('Analysis timed out. Job is still processing in the background.');
      }

      // 3. Load all evidence artifacts
      await loadExistingJob(jobId);

      // Trigger Kannada emergency popup only when confirmed prediction is Fire or Accident
      // Strict rule: Never display emergency popup for Normal
      const detectedPred = finalJob?.prediction || pollJob?.prediction;
      const detectedConf = finalJob?.confidence ?? pollJob?.confidence;
      const isEmergency = detectedPred === 'Fire' || detectedPred === 'Accident';

      if (isEmergency && jobId !== dismissedPopupJobId) {
        setDetectionPopup({
          isOpen: true,
          prediction: detectedPred,
          confidence: detectedConf,
          filename: file?.name || 'analyzed_video.mp4',
          jobId: jobId
        });
      }
    } catch (err) {
      console.error(err);
      setError(err.message || 'Analysis failed. Check backend logs.');
    } finally {
      setLoading(false);
      setProcessingStatus('');
    }
  };

  return (
    <div className="analysis-studio-page" id="page-analysis-studio">
      <div className="page-header" style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '22px', fontWeight: '700', color: 'var(--text-main)' }}>Video Analysis Studio</h2>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            Upload video files for multimodal scene classification, frame attribution, ASR, and structured description.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Quick Interactive Previews for Kannada Emergency Detection Popups */}
          <div
            style={{
              display: 'flex',
              gap: '6px',
              alignItems: 'center',
              background: 'var(--color-surface)',
              padding: '4px 8px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--color-border)'
            }}
          >
            <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              ಕನ್ನಡ Popups:
            </span>
            <button
              className="btn btn-secondary btn-sm kannada-text"
              lang="kn"
              style={{ padding: '4px 8px', fontSize: '11px' }}
              onClick={() => setDetectionPopup({
                isOpen: true,
                prediction: 'Fire',
                confidence: 0.942,
                filename: file?.name || 'fire_emergency_sample.mp4',
                jobId: 'demo-fire-01'
              })}
              id="btn-test-fire-popup"
              title="ಬೆಂಕಿ ಎಚ್ಚರಿಕೆ ಪಾಪ್-ಅಪ್ (Fire Warning)"
            >
              🔥 ಬೆಂಕಿ (Fire)
            </button>
            <button
              className="btn btn-secondary btn-sm kannada-text"
              lang="kn"
              style={{ padding: '4px 8px', fontSize: '11px' }}
              onClick={() => setDetectionPopup({
                isOpen: true,
                prediction: 'Accident',
                confidence: 0.8975,
                filename: file?.name || 'accident_incident_sample.mp4',
                jobId: 'demo-accident-01'
              })}
              id="btn-test-accident-popup"
              title="ಅಪಘಾತ ಎಚ್ಚರಿಕೆ ಪಾಪ್-ಅಪ್ (Accident Warning)"
            >
              💥 ಅಪಘಾತ (Accident)
            </button>
            <button
              className="btn btn-secondary btn-sm"
              style={{ padding: '4px 8px', fontSize: '11px' }}
              onClick={() => setDetectionPopup({
                isOpen: true,
                prediction: 'Normal',
                confidence: 0.9684,
                filename: file?.name || 'routine_traffic.mp4',
                jobId: 'demo-normal-01'
              })}
              id="btn-test-normal-popup"
              title="Normal: No emergency popup should appear"
            >
              ✅ Normal (No Alert)
            </button>
          </div>

          {(activeJob || file) && (
            <button className="btn btn-secondary btn-sm" onClick={resetState} id="btn-analyse-new">
              + New Video
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="disclaimer-banner" style={{ background: 'rgba(255, 90, 95, 0.1)', borderColor: 'rgba(255, 90, 95, 0.3)', color: '#ff6b70', marginBottom: '20px' }} id="error-message">
          <span>⚠️</span> <span>{error}</span>
        </div>
      )}

      {/* Upload Zone & Video Preview (shown when no results or when editing) */}
      {!activeJob && !loading && (
        <div className="glass-card" style={{ marginBottom: '24px', textAlign: 'center' }}>
          <div
            className={`upload-area ${isDragging ? 'active' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            id="upload-dropzone"
            style={{
              border: isDragging ? '2px dashed #6366F1' : '2px dashed rgba(255, 255, 255, 0.12)',
              borderRadius: 'var(--radius-md)',
              padding: '40px 20px',
              cursor: 'pointer',
              background: isDragging ? 'rgba(99, 102, 241, 0.08)' : 'var(--bg-card)',
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontSize: '42px', marginBottom: '12px' }}>🎬</div>
            <p style={{ fontSize: '16px', fontWeight: '600', color: 'var(--text-main)', marginBottom: '6px' }}>
              Drag & drop a video file here, or click to browse
            </p>
            <div style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
              Supports MP4, AVI, MOV, MKV, WEBM — max 100 MB
            </div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files && e.target.files[0] && handleFileSelect(e.target.files[0])}
              accept=".mp4,.avi,.mov,.mkv,.webm"
              style={{ display: 'none' }}
              id="file-input"
            />
          </div>

          {file && (
            <div style={{ marginTop: '20px', textAlign: 'left' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', background: 'var(--bg-surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ fontSize: '24px' }}>🎞️</span>
                  <div>
                    <div style={{ fontWeight: '600', color: 'var(--text-main)', fontSize: '14px' }}>{file.name}</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {(file.size / (1024 * 1024)).toFixed(2)} MB · {file.type || 'video/mp4'}
                    </div>
                  </div>
                </div>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={(e) => { e.stopPropagation(); resetState(); }}
                  id="remove-file-btn"
                >
                  ✕ Remove
                </button>
              </div>

              {videoPreviewUrl && (
                <div style={{ marginTop: '16px', background: '#000', borderRadius: 'var(--radius-sm)', overflow: 'hidden' }}>
                  <video
                    src={videoPreviewUrl}
                    controls
                    style={{ width: '100%', maxHeight: '360px', display: 'block' }}
                  />
                </div>
              )}

              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <button
                  className="btn btn-primary"
                  style={{ padding: '12px 32px', fontSize: '15px' }}
                  onClick={handleStartAnalysis}
                  disabled={loading}
                  id="analyse-btn"
                >
                  ⚡ Start Multimodal Analysis
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Loading & Progress State */}
      {loading && (
        <div className="glass-card" style={{ padding: '48px 24px', textAlign: 'center', marginBottom: '24px' }} id="loading-state">
          <div className="spinner-ring" />
          <h3 style={{ fontSize: '18px', color: 'var(--text-main)', marginBottom: '8px' }}>
            {processingStatus || 'Analyzing Video Modalities…'}
          </h3>
          <div style={{ maxWidth: '420px', margin: '14px auto 10px auto' }}>
            <div className="prob-bar-track" style={{ height: '8px' }}>
              <div
                className="prob-bar-fill fire"
                style={{ width: `${Math.max(uploadProgress, 5)}%`, transition: 'width 0.4s' }}
              />
            </div>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Extracting 12 frames, calculating mel-spectrogram, executing Whisper ASR, and computing multimodal inference.
          </div>
        </div>
      )}

      {/* Comprehensive Analysis Results View */}
      {activeJob && (
        <div className="analysis-results-section" id="result-section">
          {/* Prediction Hero Card */}
          <div className="glass-card" style={{ marginBottom: '24px', position: 'relative', overflow: 'hidden' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '16px' }}>
              <div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Multimodal Event Classification
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '4px' }}>
                  <h1 style={{ fontSize: '28px', fontWeight: '800', color: activeJob.prediction === 'Fire' ? '#FF5A5F' : activeJob.prediction === 'Accident' ? '#F59E0B' : '#10B981', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>{activeJob.prediction === 'Fire' ? '🔥' : activeJob.prediction === 'Accident' ? '💥' : '✅'}</span>
                    <span>{activeJob.prediction}</span>
                  </h1>
                  <span className="badge badge-neutral" style={{ fontSize: '13px' }}>
                    Confidence: {activeJob.confidence !== undefined ? `${(activeJob.confidence * 100).toFixed(2)}%` : '—'}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Video: <strong>{activeJob.filename}</strong> · Model: <span className="font-mono">kannada_av_scene</span> · Job ID: <span className="font-mono">{activeJob.job_id}</span>
                </div>
              </div>

              {/* Report Download & Kannada Alert Popup Trigger */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
                {(activeJob.prediction === 'Fire' || activeJob.prediction === 'Accident') && (
                  <button
                    className="btn btn-primary btn-sm kannada-text"
                    lang="kn"
                    onClick={() => setDetectionPopup({
                      isOpen: true,
                      prediction: activeJob.prediction,
                      confidence: activeJob.confidence,
                      filename: activeJob.filename,
                      jobId: activeJob.job_id
                    })}
                    id="btn-open-detection-popup"
                    title="ಕನ್ನಡ ತುರ್ತು ಎಚ್ಚರಿಕೆ ವೀಕ್ಷಿಸಿ"
                  >
                    🚨 ಕನ್ನಡ ಎಚ್ಚರಿಕೆ ({activeJob.prediction})
                  </button>
                )}
                <a
                  href={api.getReportUrl(activeJob.job_id, 'html')}
                  download={`report_${activeJob.job_id}.html`}
                  className="btn btn-secondary btn-sm"
                  id="btn-download-html"
                >
                  📄 HTML Report
                </a>
                <a
                  href={api.getReportUrl(activeJob.job_id, 'json')}
                  download={`report_${activeJob.job_id}.json`}
                  className="btn btn-secondary btn-sm"
                  id="btn-download-json"
                >
                  📊 JSON
                </a>
                <a
                  href={api.getReportUrl(activeJob.job_id, 'markdown')}
                  download={`report_${activeJob.job_id}.md`}
                  className="btn btn-secondary btn-sm"
                  id="btn-download-md"
                >
                  📝 Markdown
                </a>
              </div>
            </div>

            {/* Class Probabilities Bar Chart */}
            <div style={{ marginTop: '20px' }}>
              <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-main)', marginBottom: '8px' }}>
                Raw Softmax Activation Distribution
              </div>
              <ProbabilityBar
                probabilities={{
                  Fire: activeJob.prob_fire,
                  Accident: activeJob.prob_accident,
                  Normal: activeJob.prob_normal
                }}
                predictedClass={activeJob.prediction}
              />
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                * Scores represent direct softmax output logits from the cross-modal fusion layer (not Bayesian-calibrated probabilities).
              </div>
            </div>

            {/* Emergency Action Buttons (WhatsApp / Twilio) */}
            <EmergencyActions
              prediction={activeJob.prediction}
              confidence={activeJob.confidence}
              videoIdentifier={activeJob.filename}
              jobId={activeJob.job_id}
            />
          </div>

          {/* Evidence Exploration Tabs */}
          <div className="glass-card">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px', marginBottom: '20px' }}>
              {[
                { id: 'frames', label: '12 Sampled Frames', icon: '🎞️', count: frames.length },
                { id: 'attribution', label: 'Influential Frame Attribution', icon: '🎯', count: attributionsData?.attributions?.length },
                { id: 'audio', label: 'Acoustic & Mel Spectrogram', icon: '🎵', badge: audioData?.has_audio_stream ? 'Present' : 'Silent' },
                { id: 'speech', label: 'Kannada/English Transcript', icon: '🎙️', count: transcripts.length || (activeJob.transcript ? 1 : 0) },
                { id: 'description', label: '6-Section Scene Description', icon: '📝', badge: 'Evidence-Grounded' }
              ].map(tab => (
                <button
                  key={tab.id}
                  className={`btn ${activeTab === tab.id ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => setActiveTab(tab.id)}
                  id={`tab-${tab.id}`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                  {tab.count !== undefined && <span className="font-mono">({tab.count})</span>}
                  {tab.badge && <span className="badge badge-neutral" style={{ fontSize: '10px' }}>{tab.badge}</span>}
                </button>
              ))}
            </div>

            {/* Tab 1: 12 Extracted Frames */}
            {activeTab === 'frames' && (
              <FrameViewer
                frames={frames}
                attributions={attributionsData?.attributions}
                filmstripUrl={api.getFilmstripUrl(activeJob.job_id)}
              />
            )}

            {/* Tab 2: Influential Frame Attribution */}
            {activeTab === 'attribution' && (
              <AttributionViewer attributionsData={attributionsData} />
            )}

            {/* Tab 3: Acoustic & Mel Spectrogram */}
            {activeTab === 'audio' && (
              <AudioSpectrogramViewer
                audioData={audioData}
                spectrogramUrl={api.getSpectrogramUrl(activeJob.job_id)}
              />
            )}

            {/* Tab 4: Speech Transcript */}
            {activeTab === 'speech' && (
              <TranscriptViewer
                transcripts={transcripts}
                rawTranscript={activeJob.transcript}
              />
            )}

            {/* Tab 5: 6-Section Evidence-Grounded Description */}
            {activeTab === 'description' && (
              <SceneDescriptionViewer
                descriptionData={descriptionData || activeJob.structured_description}
                formattedText={activeJob.description}
              />
            )}
          </div>
        </div>
      )}

      {/* Kannada Emergency Detection Popup Modal (Fire & Accident only) */}
      <DetectionPopup
        isOpen={detectionPopup.isOpen}
        onClose={() => {
          if (detectionPopup.jobId) {
            setDismissedPopupJobId(detectionPopup.jobId);
          }
          setDetectionPopup(prev => ({ ...prev, isOpen: false }));
        }}
        prediction={detectionPopup.prediction}
        confidence={detectionPopup.confidence}
        filename={detectionPopup.filename}
        jobId={detectionPopup.jobId}
      />
    </div>
  );
}
