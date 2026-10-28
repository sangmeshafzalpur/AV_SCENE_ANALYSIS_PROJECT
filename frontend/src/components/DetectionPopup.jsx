import React, { useEffect, useState } from 'react';
import { api } from '../api';

/**
 * Kannada Emergency Detection Popup Component
 * Displays a prominent emergency modal in Kannada when video analysis returns Fire or Accident.
 * Strict Rule: Never displays an emergency popup for Normal.
 *
 * For Fire: "ಎಚ್ಚರಿಕೆ! ಬೆಂಕಿ ಅವಘಡ ಪತ್ತೆಯಾಗಿದೆ!" (Warning! Fire detected!)
 * For Accident: "ಎಚ್ಚರಿಕೆ! ಅಪಘಾತದ ದೃಶ್ಯ ಪತ್ತೆಯಾಗಿದೆ!" (Warning! Accident scene detected!)
 * Dismissal Button: "ಮುಚ್ಚಿ" (Close)
 */
export function DetectionPopup({
  isOpen,
  onClose,
  prediction,
  confidence,
  filename,
  jobId
}) {
  const [waLoading, setWaLoading] = useState(false);
  const [waSent, setWaSent] = useState(false);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Strict Rule: Trigger ONLY for Fire or Accident. Never display for Normal or missing prediction.
  if (!isOpen || !prediction) return null;

  const normalizedPred = String(prediction).trim();
  const isFire = prediction === 'Fire' || normalizedPred.toLowerCase() === 'fire';
  const isAccident = prediction === 'Accident' || normalizedPred.toLowerCase() === 'accident';
  const isNormal = prediction === 'Normal' || normalizedPred.toLowerCase() === 'normal';

  // Do not display emergency popup for Normal (Scene Classified as Normal is excluded from emergency popup)
  if (isNormal || (!isFire && !isAccident)) {
    return null;
  }

  const confPercent = confidence !== undefined && confidence !== null
    ? `${(confidence * 100).toFixed(2)}%`
    : null;

  // Handle WhatsApp alert dispatch for emergencies
  const handleTriggerWhatsApp = async () => {
    setWaLoading(true);
    try {
      const data = await api.getWhatsAppLink(prediction, confidence || 0.95, filename || 'video_sample.mp4');
      if (data?.url) {
        window.open(data.url, '_blank', 'noopener,noreferrer');
        setWaSent(true);
      }
    } catch (err) {
      console.error('Failed to get WhatsApp alert link:', err);
    } finally {
      setWaLoading(false);
    }
  };

  // Kannada Emergency Alert Configuration
  // For Fire: "ಎಚ್ಚರಿಕೆ! ಬೆಂಕಿ ಅವಘಡ ಪತ್ತೆಯಾಗಿದೆ!" (CRITICAL ALERT: Fire Detected!)
  // For Accident: "ಎಚ್ಚರಿಕೆ! ಅಪಘಾತದ ದೃಶ್ಯ ಪತ್ತೆಯಾಗಿದೆ!" (INCIDENT ALERT: Accident Detected!)
  const config = isFire
    ? {
        icon: '🔥',
        kannadaMessage: 'ಎಚ್ಚರಿಕೆ! ಬೆಂಕಿ ಅವಘಡ ಪತ್ತೆಯಾಗಿದೆ!',
        englishMeaning: 'Warning! Fire detected!',
        title: 'CRITICAL ALERT: Fire Detected!',
        badgeClass: 'badge-fire',
        badgeText: 'ಬೆಂಕಿ ಅವಘಡ (FIRE)',
        borderColor: '#E43D12',
        glowColor: 'rgba(228, 61, 18, 0.38)',
        headingColor: '#E43D12',
        subtext: 'ವಿಡಿಯೋ ದೃಶ್ಯದಲ್ಲಿ ಬೆಂಕಿ ಅಥವಾ ಹೊಗೆ ಪತ್ತೆಯಾಗಿದೆ. ತಕ್ಷಣ ತುರ್ತು ಕ್ರಮ ಕೈಗೊಳ್ಳಿ.',
        primaryActionText: '📱 WhatsApp ಎಚ್ಚರಿಕೆ ರವಾನಿಸಿ',
        primaryAction: handleTriggerWhatsApp
      }
    : {
        icon: '💥',
        kannadaMessage: 'ಎಚ್ಚರಿಕೆ! ಅಪಘಾತದ ದೃಶ್ಯ ಪತ್ತೆಯಾಗಿದೆ!',
        englishMeaning: 'Warning! Accident scene detected!',
        title: 'INCIDENT ALERT: Accident Detected!',
        badgeClass: 'badge-accident',
        badgeText: 'ಅಪಘಾತದ ದೃಶ್ಯ (ACCIDENT)',
        borderColor: '#EFB11D',
        glowColor: 'rgba(239, 177, 29, 0.38)',
        headingColor: '#8B6200',
        subtext: 'ವಿಡಿಯೋ ದೃಶ್ಯದಲ್ಲಿ ವಾಹನ ಅಥವಾ ರಸ್ತೆ ಅಪಘಾತ ಪತ್ತೆಯಾಗಿದೆ. ತುರ್ತು ನೆರವು ಪರಿಶೀಲಿಸಿ.',
        primaryActionText: '📱 WhatsApp ಮಾಹಿತಿ ರವಾನಿಸಿ',
        primaryAction: handleTriggerWhatsApp
      };

  return (
    <div
      className="modal-overlay"
      id="kannada-emergency-popup-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="kannada-popup-heading"
      style={{
        zIndex: 120,
        backgroundColor: 'rgba(41, 37, 34, 0.65)',
        backdropFilter: 'blur(6px)',
        animation: 'modalIn 0.2s ease-out'
      }}
    >
      <div
        className="modal-content"
        id="kannada-emergency-popup-content"
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: '540px',
          width: '92%',
          border: `2.5px solid ${config.borderColor}`,
          boxShadow: `0 20px 48px rgba(41, 37, 34, 0.28), 0 0 28px ${config.glowColor}`,
          borderRadius: 'var(--radius-md)',
          background: '#FFFFFF',
          overflow: 'hidden'
        }}
      >
        {/* Top Emergency Visual Stripe */}
        <div
          style={{
            height: '6px',
            background: isFire
              ? 'linear-gradient(90deg, #E43D12, #D6536D)'
              : 'linear-gradient(90deg, #EFB11D, #E43D12)'
          }}
        />

        {/* Modal Header */}
        <div
          className="modal-header"
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid var(--color-border)',
            background: isFire ? 'rgba(228, 61, 18, 0.08)' : 'rgba(239, 177, 29, 0.12)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '26px' }} aria-hidden="true">{config.icon}</span>
            <span className={`badge ${config.badgeClass}`} style={{ fontSize: '11px', fontWeight: 700 }}>
              {config.badgeText}
            </span>
          </div>

          <button
            onClick={onClose}
            aria-label="ಮುಚ್ಚಿ (Close)"
            id="btn-close-kannada-popup"
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '20px',
              color: 'var(--color-text-secondary)',
              cursor: 'pointer',
              padding: '4px 8px',
              borderRadius: 'var(--radius-xs)',
              lineHeight: 1
            }}
          >
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body" style={{ padding: '24px 22px' }}>
          {/* Prominent Kannada Message */}
          <h2
            id="kannada-popup-heading"
            className="kannada-text"
            lang="kn"
            style={{
              fontSize: '23px',
              fontWeight: 800,
              color: config.headingColor,
              margin: '0 0 8px 0',
              lineHeight: 1.5,
              letterSpacing: '0.01em'
            }}
          >
            {config.kannadaMessage}
          </h2>

          {/* English Meaning Reference */}
          <div
            style={{
              fontSize: '13px',
              fontWeight: 600,
              color: 'var(--color-text-secondary)',
              marginBottom: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <span>({config.englishMeaning})</span>
          </div>

          <p
            className="kannada-text"
            lang="kn"
            style={{
              fontSize: '13.5px',
              lineHeight: 1.6,
              color: 'var(--color-text-primary)',
              marginBottom: '18px'
            }}
          >
            {config.subtext}
          </p>

          {/* Metadata Summary */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
              gap: '10px',
              padding: '12px 14px',
              background: '#F5F3ED',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--color-border)',
              marginBottom: '20px'
            }}
          >
            <div>
              <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>
                ಪತ್ತೆಯಾದ ವರ್ಗ (Class)
              </div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: config.headingColor, marginTop: '2px' }}>
                {prediction}
              </div>
            </div>

            {confPercent && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>
                  ನಿಖರತೆ (Confidence)
                </div>
                <div className="font-mono" style={{ fontSize: '15px', fontWeight: 700, color: 'var(--color-text-primary)', marginTop: '2px' }}>
                  {confPercent}
                </div>
              </div>
            )}

            {filename && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>
                  ವಿಡಿಯೋ (Video)
                </div>
                <div
                  style={{
                    fontSize: '12.5px',
                    fontWeight: 600,
                    color: 'var(--color-text-primary)',
                    marginTop: '2px',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}
                  title={filename}
                >
                  {filename}
                </div>
              </div>
            )}

            {jobId && (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', textTransform: 'uppercase' }}>
                  ಉಲ್ಲೇಖ (Job ID)
                </div>
                <div
                  className="font-mono"
                  style={{
                    fontSize: '11.5px',
                    fontWeight: 600,
                    color: 'var(--color-text-primary)',
                    marginTop: '2px',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}
                  title={jobId}
                >
                  {jobId}
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons with Kannada Dismissal Button */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'flex-end', alignItems: 'center' }}>
            {config.primaryAction && (
              <button
                className={isFire ? 'btn btn-primary btn-sm' : 'btn btn-warning btn-sm'}
                onClick={config.primaryAction}
                disabled={waLoading || waSent}
                id="btn-kannada-popup-whatsapp"
              >
                {waLoading ? 'ಸಂಪರ್ಕಿಸಲಾಗುತ್ತಿದೆ...' : waSent ? '✓ ರವಾನಿಸಲಾಗಿದೆ' : config.primaryActionText}
              </button>
            )}

            {/* Mandatory Kannada Dismissal Button labeled "ಮುಚ್ಚಿ" */}
            <button
              className="btn btn-secondary btn-sm kannada-text"
              lang="kn"
              onClick={onClose}
              id="btn-kannada-popup-dismiss"
              style={{ fontWeight: 700, minWidth: '85px', fontSize: '13px' }}
            >
              ಮುಚ್ಚಿ
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DetectionPopup;
