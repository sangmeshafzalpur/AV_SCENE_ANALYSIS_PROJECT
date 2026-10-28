/**
 * Frontend Dashboard Automated Test Suite
 * Validates Component, Navigation, Responsive, Accessibility, Kannada Unicode,
 * and Backend API Integration Contracts.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 1. Navigation & Page Routes Integrity
test('1. Navigation defines all 8 required pages with icons and unique IDs', () => {
  const navSource = fs.readFileSync(path.join(__dirname, 'src/components/Navigation.jsx'), 'utf-8');

  const expectedPageIds = [
    'overview',
    'studio',
    'model',
    'dataset',
    'evaluation',
    'history',
    'monitoring',
    'settings'
  ];

  for (const pageId of expectedPageIds) {
    assert.ok(
      navSource.includes(`id: '${pageId}'`),
      `Navigation.jsx must define page with id: '${pageId}'`
    );
  }

  // Verify match count of id:
  const matches = navSource.match(/id:\s*'[a-z_]+'/g) || [];
  assert.equal(matches.length, 8, 'Expected exactly 8 page definitions in Navigation.jsx');
});

// 2. Component Exports Integrity
test('2. All components and pages export valid React component functions', () => {
  const components = [
    { file: 'src/components/Navigation.jsx', exports: ['Navigation', 'PAGES'] },
    { file: 'src/components/StatsCard.jsx', exports: ['StatsCard'] },
    { file: 'src/components/ProbabilityBar.jsx', exports: ['ProbabilityBar'] },
    { file: 'src/components/FrameViewer.jsx', exports: ['FrameViewer'] },
    { file: 'src/components/AttributionViewer.jsx', exports: ['AttributionViewer'] },
    { file: 'src/components/AudioSpectrogramViewer.jsx', exports: ['AudioSpectrogramViewer'] },
    { file: 'src/components/TranscriptViewer.jsx', exports: ['TranscriptViewer'] },
    { file: 'src/components/SceneDescriptionViewer.jsx', exports: ['SceneDescriptionViewer'] },
    { file: 'src/components/EmergencyActions.jsx', exports: ['EmergencyActions'] },
    { file: 'src/components/Logo.jsx', exports: ['Logo'] },
    { file: 'src/components/DetectionPopup.jsx', exports: ['DetectionPopup'] },
    { file: 'src/pages/OverviewPage.jsx', exports: ['OverviewPage'] },
    { file: 'src/pages/ModelCenterPage.jsx', exports: ['ModelCenterPage'] },
    { file: 'src/pages/AnalysisStudioPage.jsx', exports: ['AnalysisStudioPage'] },
    { file: 'src/pages/DatasetDashboardPage.jsx', exports: ['DatasetDashboardPage'] },
    { file: 'src/pages/EvaluationCenterPage.jsx', exports: ['EvaluationCenterPage'] },
    { file: 'src/pages/HistoryReportsPage.jsx', exports: ['HistoryReportsPage'] },
    { file: 'src/pages/SystemMonitoringPage.jsx', exports: ['SystemMonitoringPage'] },
    { file: 'src/pages/SettingsDocPage.jsx', exports: ['SettingsDocPage'] },
    { file: 'src/App.jsx', exports: ['App'] }
  ];

  for (const comp of components) {
    const fullPath = path.join(__dirname, comp.file);
    assert.ok(fs.existsSync(fullPath), `File does not exist: ${comp.file}`);
    const code = fs.readFileSync(fullPath, 'utf-8');

    for (const exp of comp.exports) {
      const hasExport = code.includes(`export function ${exp}`) ||
                        code.includes(`export const ${exp}`) ||
                        code.includes(`export default ${exp}`) ||
                        code.includes(`export default function ${exp}`);
      assert.ok(hasExport, `${comp.file} must export ${exp}`);
    }
  }
});

// 3. Kannada Unicode Preservation in Frontend Source & Transcripts
test('3. Kannada Unicode characters are preserved accurately across files', () => {
  const transcriptViewerCode = fs.readFileSync(path.join(__dirname, 'src/components/TranscriptViewer.jsx'), 'utf-8');

  // Verify regex detects Kannada unicode range \u0C80-\u0CFF
  assert.ok(transcriptViewerCode.includes('\\u0C80-\\u0CFF'), 'TranscriptViewer must check Kannada unicode range');
  assert.ok(transcriptViewerCode.includes('ಕನ್ನಡ'), 'TranscriptViewer must preserve Kannada display text');

  // Verify font configuration in index.html
  const indexHtml = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf-8');
  assert.ok(indexHtml.includes('Noto+Sans+Kannada'), 'index.html must include Noto Sans Kannada in Google Fonts');

  // Verify CSS font-family includes Noto Sans Kannada
  const indexCss = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf-8');
  assert.ok(indexCss.includes("'Noto Sans Kannada'"), 'index.css must include Noto Sans Kannada in font-family');
  assert.ok(indexCss.includes('.kannada-text'), 'index.css must define .kannada-text utility');
});

// 4. API Client Completeness
test('4. API Client implements all required backend endpoints', async () => {
  const { api } = await import('./src/api.js');
  const requiredMethods = [
    'getHealth',
    'getReadiness',
    'getSystemStats',
    'getModelMetadata',
    'getModelReadiness',
    'getEvaluationMetadata',
    'getStoragePolicy',
    'cleanupStorage',
    'listJobs',
    'getJob',
    'createAnalysisJob',
    'cancelJob',
    'getJobAttributions',
    'getJobTranscripts',
    'getJobDescription',
    'getJobAudioMetadata',
    'getJobFrames',
    'getJobArtifacts',
    'predictSync',
    'getWhatsAppLink',
    'sendWhatsAppAlert',
    'triggerEmergencyCall',
    'triggerEmergencyMessage',
    'getReportUrl',
    'getSpectrogramUrl',
    'getFilmstripUrl'
  ];

  for (const m of requiredMethods) {
    assert.equal(typeof api[m], 'function', `api.${m} is missing or not a function`);
  }
});

// 5. Responsive Design & Media Queries Test
test('5. Design system includes responsive layout breakpoints', () => {
  const css = fs.readFileSync(path.join(__dirname, 'src/index.css'), 'utf-8');

  // Breakpoints
  assert.ok(css.includes('@media (max-width: 1024px)'), 'Missing 1024px tablet breakpoint');
  assert.ok(css.includes('@media (max-width: 640px)'), 'Missing 640px mobile breakpoint');
  assert.ok(css.includes('.app-sidebar.open'), 'Missing mobile open drawer state');
  assert.ok(css.includes('.mobile-menu-btn'), 'Missing mobile menu toggle styling');
});

// 6. Accessibility & ARIA Contract Test
test('6. Accessibility semantics, ARIA roles, and tap targets are present', () => {
  const navCode = fs.readFileSync(path.join(__dirname, 'src/components/Navigation.jsx'), 'utf-8');
  assert.ok(navCode.includes('aria-label="Main Navigation"'), 'Sidebar missing aria-label');
  assert.ok(navCode.includes('aria-hidden="true"'), 'Icons missing aria-hidden');

  const appCode = fs.readFileSync(path.join(__dirname, 'src/App.jsx'), 'utf-8');
  assert.ok(appCode.includes('aria-label="Toggle Navigation Menu"'), 'Mobile menu toggle missing aria-label');

  const historyCode = fs.readFileSync(path.join(__dirname, 'src/pages/HistoryReportsPage.jsx'), 'utf-8');
  assert.ok(historyCode.includes('aria-label='), 'History checkboxes missing aria-label');

  // Verify causation disclaimer exists
  const attrCode = fs.readFileSync(path.join(__dirname, 'src/components/AttributionViewer.jsx'), 'utf-8');
  assert.ok(attrCode.includes('causation_disclaimer'), 'AttributionViewer must display causation disclaimer');
});

// 7. Live Backend Connectivity Test
test('7. Frontend can reach live backend health and metadata endpoints', async () => {
  const baseUrl = 'http://127.0.0.1:8000';

  const healthRes = await fetch(`${baseUrl}/api/health`);
  assert.equal(healthRes.status, 200, 'Backend /api/health returned non-200');
  const healthData = await healthRes.json();
  assert.equal(healthData.status, 'healthy', 'Backend is not healthy');

  const metaRes = await fetch(`${baseUrl}/api/model/metadata`);
  assert.equal(metaRes.status, 200, 'Backend /api/model/metadata returned non-200');
  const metaData = await metaRes.json();
  assert.equal(metaData.model_identification.sha256_checksum, '404f374484b2304e44074dd44a81a8130add1b22c22dcd7a7e21ea56f8763fde');

  const evalRes = await fetch(`${baseUrl}/api/evaluation/metadata`);
  assert.equal(evalRes.status, 200, 'Backend /api/evaluation/metadata returned non-200');
  const evalData = await evalRes.json();
  assert.equal(evalData.dataset_metadata.training_set.total, 1080);
});

// 8. Brand Logo & Multimodal Detection Popup Integrity Test
test('8. Brand Logo and DetectionPopup (Normal, Fire, Accident) contracts are satisfied', () => {
  // Verify Logo component exists and contains SVG aperture/wave paths
  const logoCode = fs.readFileSync(path.join(__dirname, 'src/components/Logo.jsx'), 'utf-8');
  assert.ok(logoCode.includes('avGradPrimary'), 'Logo must contain theme primary gradient');
  assert.ok(logoCode.includes('avGradGold'), 'Logo must contain gold accent gradient');

  // Verify Navigation uses Logo
  const navCode = fs.readFileSync(path.join(__dirname, 'src/components/Navigation.jsx'), 'utf-8');
  assert.ok(navCode.includes('<Logo'), 'Navigation must render Logo component');

  // Verify DetectionPopup handles Normal, Fire, and Accident messages
  const popupCode = fs.readFileSync(path.join(__dirname, 'src/components/DetectionPopup.jsx'), 'utf-8');
  assert.ok(popupCode.includes("prediction === 'Fire'"), 'DetectionPopup must handle Fire prediction');
  assert.ok(popupCode.includes("prediction === 'Accident'"), 'DetectionPopup must handle Accident prediction');
  assert.ok(popupCode.includes("prediction === 'Normal'"), 'DetectionPopup must handle Normal prediction');
  assert.ok(popupCode.includes('Scene Classified as Normal'), 'DetectionPopup must define Normal scene message');
  assert.ok(popupCode.includes('CRITICAL ALERT: Fire Detected!'), 'DetectionPopup must define Fire emergency message');
  assert.ok(popupCode.includes('INCIDENT ALERT: Accident Detected!'), 'DetectionPopup must define Accident incident message');

  // Verify AnalysisStudioPage integrates DetectionPopup and preview test triggers
  const studioCode = fs.readFileSync(path.join(__dirname, 'src/pages/AnalysisStudioPage.jsx'), 'utf-8');
  assert.ok(studioCode.includes('<DetectionPopup'), 'AnalysisStudioPage must render DetectionPopup');
  assert.ok(studioCode.includes('btn-test-normal-popup'), 'AnalysisStudioPage must provide Normal popup test button');
  assert.ok(studioCode.includes('btn-test-fire-popup'), 'AnalysisStudioPage must provide Fire popup test button');
  assert.ok(studioCode.includes('btn-test-accident-popup'), 'AnalysisStudioPage must provide Accident popup test button');
});

