import React from 'react';

const CLASS_EMOJI = { Fire: '🔥', Accident: '💥', Normal: '✅' };
const CLASS_KEY = { Fire: 'fire', Accident: 'accident', Normal: 'normal' };

export function ProbabilityBar({ probabilities, predictedClass }) {
  if (!probabilities) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '13px', padding: '8px 0' }}>
        Probabilities not available
      </div>
    );
  }

  const classes = ['Fire', 'Accident', 'Normal'];

  const formatPercent = (val) => {
    if (val === undefined || val === null) return '0.00%';
    return `${(val * 100).toFixed(2)}%`;
  };

  return (
    <div className="probabilities-wrapper" style={{ margin: '14px 0' }}>
      {classes.map(cls => {
        const val = probabilities[cls] ?? 0;
        const key = CLASS_KEY[cls] || 'normal';
        const isPredicted = cls === predictedClass;

        return (
          <div className="prob-row" key={cls}>
            <div className="prob-label">
              <span>{CLASS_EMOJI[cls]}</span>
              <span>{cls}</span>
            </div>
            <div className="prob-bar-track">
              <div
                className={`prob-bar-fill ${key} ${isPredicted ? 'highlight' : ''}`}
                style={{ width: `${Math.min(Math.max(val * 100, 1), 100)}%` }}
              />
            </div>
            <div className="prob-value" style={isPredicted ? { fontWeight: '700', color: 'var(--text-main)' } : undefined}>
              {formatPercent(val)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
