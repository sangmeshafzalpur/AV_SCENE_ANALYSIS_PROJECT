import React from 'react';

export function StatsCard({ title, value, subtext, icon, trend, color, id }) {
  return (
    <div className="kpi-card" id={id}>
      <div className="kpi-header">
        <span>{title}</span>
        {icon && <span style={{ fontSize: '18px' }}>{icon}</span>}
      </div>
      <div className="kpi-val" style={color ? { color } : undefined}>
        {value !== undefined && value !== null ? value : <span style={{ fontSize: '16px', color: 'var(--text-muted)' }}>Not available</span>}
      </div>
      {subtext && (
        <div className="kpi-subtext">
          {trend && <span>{trend}</span>}
          <span>{subtext}</span>
        </div>
      )}
    </div>
  );
}
