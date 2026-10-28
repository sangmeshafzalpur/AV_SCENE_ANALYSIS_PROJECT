import React from 'react';

/**
 * AV Scene Analysis Brand Logo
 * Vector emblem fusing visual camera aperture, audio spectrogram frequencies,
 * and central AI intelligence node in the 5-color theme palette.
 */
export function Logo({ size = 36, showText = false, className = '' }) {
  return (
    <div
      className={`av-brand-logo ${className}`}
      style={{ display: 'inline-flex', alignItems: 'center', gap: '10px' }}
      title="AV Scene Analysis — Multimodal AI"
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          flexShrink: 0,
          filter: 'drop-shadow(0 3px 10px rgba(228, 61, 18, 0.32))'
        }}
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="avGradPrimary" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#E43D12" />
            <stop offset="55%" stopColor="#D6536D" />
            <stop offset="100%" stopColor="#FFA2B6" />
          </linearGradient>

          <linearGradient id="avGradGold" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#EFB11D" />
            <stop offset="100%" stopColor="#E43D12" />
          </linearGradient>

          <linearGradient id="avGradShield" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#2E2824" />
            <stop offset="100%" stopColor="#1B1715" />
          </linearGradient>

          <filter id="avGlow" x="-25%" y="-25%" width="150%" height="150%">
            <feGaussianBlur stdDeviation="2.5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Outer Squircle Container with Primary Border */}
        <rect
          x="6"
          y="6"
          width="88"
          height="88"
          rx="22"
          fill="url(#avGradShield)"
          stroke="url(#avGradPrimary)"
          strokeWidth="3.5"
        />

        {/* Optical Sensor Aperture Ring */}
        <circle
          cx="50"
          cy="50"
          r="32"
          stroke="url(#avGradGold)"
          strokeWidth="1.8"
          strokeDasharray="5 3"
          opacity="0.85"
        />
        <circle
          cx="50"
          cy="50"
          r="26"
          fill="rgba(228, 61, 18, 0.08)"
          stroke="rgba(235, 233, 225, 0.2)"
          strokeWidth="1"
        />

        {/* Visual Frame Bracket Guides */}
        <path
          d="M32 27 L25 27 L25 34"
          stroke="#FFA2B6"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M68 27 L75 27 L75 34"
          stroke="#FFA2B6"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M32 73 L25 73 L25 66"
          stroke="#FFA2B6"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M68 73 L75 73 L75 66"
          stroke="#FFA2B6"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Audio Waveform Equalizer Bars */}
        <line x1="33" y1="44" x2="33" y2="56" stroke="#EFB11D" strokeWidth="3" strokeLinecap="round" />
        <line x1="39" y1="36" x2="39" y2="64" stroke="url(#avGradPrimary)" strokeWidth="3.5" strokeLinecap="round" />
        <line x1="45" y1="31" x2="45" y2="69" stroke="#E43D12" strokeWidth="4" strokeLinecap="round" />
        <line x1="55" y1="31" x2="55" y2="69" stroke="#D6536D" strokeWidth="4" strokeLinecap="round" />
        <line x1="61" y1="36" x2="61" y2="64" stroke="url(#avGradPrimary)" strokeWidth="3.5" strokeLinecap="round" />
        <line x1="67" y1="44" x2="67" y2="56" stroke="#FFA2B6" strokeWidth="3" strokeLinecap="round" />

        {/* Central Intelligent Fusion Node */}
        <circle cx="50" cy="50" r="5" fill="#EFB11D" filter="url(#avGlow)" />
        <circle cx="50" cy="50" r="2.2" fill="#FFFFFF" />

        {/* Multimodal Spark Highlight */}
        <path
          d="M74 16 L76 22 L82 24 L76 26 L74 32 L72 26 L66 24 L72 22 Z"
          fill="#EFB11D"
          filter="url(#avGlow)"
        />
      </svg>

      {showText && (
        <div className="brand-text">
          <h1 style={{ margin: 0, fontSize: '15px', fontWeight: 700, color: 'var(--color-text-primary)' }}>
            AV Scene Analysis
          </h1>
          <div
            className="tagline"
            style={{
              fontSize: '11px',
              color: 'var(--color-text-secondary)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}
          >
            Kannada & English Multimodal AI
          </div>
        </div>
      )}
    </div>
  );
}

export default Logo;
