import React from 'react';
import { Logo } from './Logo';

export const PAGES = [
  { id: 'overview', label: 'Project Overview', icon: '📊' },
  { id: 'studio', label: 'Video Analysis Studio', icon: '🎬' },
  { id: 'model', label: 'Model Info Center', icon: '🧠' },
  { id: 'dataset', label: 'Dataset Dashboard', icon: '📁' },
  { id: 'evaluation', label: 'Evaluation Center', icon: '📈' },
  { id: 'history', label: 'History & Reports', icon: '📜' },
  { id: 'monitoring', label: 'Alerts & Monitoring', icon: '🚨' },
  { id: 'settings', label: 'Settings & Docs', icon: '⚙️' },
];

export function Navigation({ activePage, setActivePage, isMobileOpen, setIsMobileOpen, systemHealth }) {
  const isHealthy = systemHealth?.healthy ?? true;

  return (
    <>
      <aside className={`app-sidebar ${isMobileOpen ? 'open' : ''}`} id="main-sidebar">
        <div className="sidebar-brand">
          <Logo size={40} />
          <div className="brand-text">
            <h1>AV Scene Analysis</h1>
            <div className="tagline">Kannada & English Multimodal AI</div>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Main Navigation">
          {PAGES.map(page => (
            <button
              key={page.id}
              className={`nav-item ${activePage === page.id ? 'active' : ''}`}
              onClick={() => {
                setActivePage(page.id);
                setIsMobileOpen(false);
              }}
              id={`nav-link-${page.id}`}
            >
              <span className="nav-icon" aria-hidden="true">{page.icon}</span>
              <span>{page.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="status-pill" title="Live system readiness status">
            <span className={`status-dot ${isHealthy ? 'green' : 'red'}`} />
            <span>{isHealthy ? 'System Operational' : 'Service Degraded'}</span>
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px' }}>
            Model: <span className="font-mono">kannada_av_scene</span>
          </div>
        </div>
      </aside>

      {isMobileOpen && (
        <div
          className="modal-overlay"
          style={{ zIndex: 35 }}
          onClick={() => setIsMobileOpen(false)}
          aria-hidden="true"
        />
      )}
    </>
  );
}
