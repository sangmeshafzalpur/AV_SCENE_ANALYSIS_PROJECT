import React, { useState, useEffect } from 'react';
import { api } from './api';
import { Navigation, PAGES } from './components/Navigation';
import { OverviewPage } from './pages/OverviewPage';
import { ModelCenterPage } from './pages/ModelCenterPage';
import { AnalysisStudioPage } from './pages/AnalysisStudioPage';
import { DatasetDashboardPage } from './pages/DatasetDashboardPage';
import { EvaluationCenterPage } from './pages/EvaluationCenterPage';
import { HistoryReportsPage } from './pages/HistoryReportsPage';
import { SystemMonitoringPage } from './pages/SystemMonitoringPage';
import { SettingsDocPage } from './pages/SettingsDocPage';

export function App() {
  const [activePage, setActivePage] = useState('overview');
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);
  const [studioJobId, setStudioJobId] = useState(null);

  // Poll health on mount and periodically
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await api.getHealth();
        setSystemHealth(res);
      } catch {
        setSystemHealth({ healthy: false, status: 'unreachable' });
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigateToJob = (jobId) => {
    setStudioJobId(jobId);
    setActivePage('studio');
  };

  const handleClearStudioJob = () => {
    setStudioJobId(null);
  };

  const currentPageObj = PAGES.find(p => p.id === activePage) || PAGES[0];

  return (
    <div className="app-layout">
      {/* Sidebar Navigation */}
      <Navigation
        activePage={activePage}
        setActivePage={setActivePage}
        isMobileOpen={isMobileOpen}
        setIsMobileOpen={setIsMobileOpen}
        systemHealth={systemHealth}
      />

      {/* Main Content Area */}
      <main className="app-main" id="main-content-area">
        {/* Top Header Bar */}
        <header className="top-bar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <button
              className="mobile-menu-btn"
              onClick={() => setIsMobileOpen(!isMobileOpen)}
              aria-label="Toggle Navigation Menu"
              id="btn-mobile-menu"
            >
              ☰
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '18px' }}>{currentPageObj.icon}</span>
              <h1 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-main)', margin: 0 }}>
                {currentPageObj.label}
              </h1>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div className="status-pill" id="header-status-pill">
              <span className={`status-dot ${systemHealth?.healthy ? 'green' : 'red'}`} />
              <span style={{ fontSize: '12px' }}>
                {systemHealth?.healthy ? 'API & Models Ready' : 'Degraded'}
              </span>
            </div>
            {activePage !== 'studio' && (
              <button
                className="btn btn-primary btn-sm"
                onClick={() => setActivePage('studio')}
                id="btn-quick-analyse"
              >
                + New Analysis
              </button>
            )}
          </div>
        </header>

        {/* Dynamic Page Views */}
        <div className="page-content">
          {activePage === 'overview' && (
            <OverviewPage
              onNavigateToJob={handleNavigateToJob}
              onNavigateToPage={setActivePage}
            />
          )}

          {activePage === 'studio' && (
            <AnalysisStudioPage
              initialJobId={studioJobId}
              onClearInitialJob={handleClearStudioJob}
            />
          )}

          {activePage === 'model' && <ModelCenterPage />}

          {activePage === 'dataset' && <DatasetDashboardPage />}

          {activePage === 'evaluation' && <EvaluationCenterPage />}

          {activePage === 'history' && (
            <HistoryReportsPage onInspectJob={handleNavigateToJob} />
          )}

          {activePage === 'monitoring' && <SystemMonitoringPage />}

          {activePage === 'settings' && <SettingsDocPage />}
        </div>
      </main>
    </div>
  );
}

export default App;
