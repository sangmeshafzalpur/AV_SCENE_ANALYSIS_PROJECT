import React, { useState } from 'react';
import { api } from '../api';

export function EmergencyActions({ prediction, confidence, videoIdentifier, jobId }) {
  const [waSending, setWaSending] = useState(false);
  const [waStatus, setWaStatus] = useState(null); // 'success' | 'error' | null
  const [twilioCallSending, setTwilioCallSending] = useState(false);
  const [twilioCallStatus, setTwilioCallStatus] = useState(null);
  const [twilioMsgSending, setTwilioMsgSending] = useState(false);
  const [twilioMsgStatus, setTwilioMsgStatus] = useState(null);

  const isEmergency = prediction === 'Fire' || prediction === 'Accident';

  const handleWhatsApp = async () => {
    if (!prediction) return;
    const waWindow = window.open('about:blank', '_blank');
    setWaSending(true);
    setWaStatus(null);

    try {
      const data = await api.getWhatsAppLink(prediction, confidence, videoIdentifier || 'video_sample.mp4');
      if (waWindow) {
        waWindow.location.href = data.url;
      } else {
        window.location.href = data.url;
      }
      setWaStatus('success');
    } catch (err) {
      console.error(err);
      if (waWindow) waWindow.close();
      setWaStatus('error');
    } finally {
      setWaSending(false);
    }
  };

  const handleTwilio = async (actionType) => {
    if (!prediction) return;
    const eventId = jobId || `ev_${Math.random().toString(36).substring(7)}`;

    if (actionType === 'call') {
      setTwilioCallSending(true);
      setTwilioCallStatus(null);
    } else {
      setTwilioMsgSending(true);
      setTwilioMsgStatus(null);
    }

    try {
      if (actionType === 'call') {
        await api.triggerEmergencyCall(prediction, confidence, videoIdentifier || 'video_sample.mp4', eventId);
        setTwilioCallStatus('success');
      } else {
        await api.triggerEmergencyMessage(prediction, confidence, videoIdentifier || 'video_sample.mp4', eventId);
        setTwilioMsgStatus('success');
      }
    } catch (err) {
      console.error(err);
      if (actionType === 'call') setTwilioCallStatus('error');
      else setTwilioMsgStatus('error');
    } finally {
      if (actionType === 'call') setTwilioCallSending(false);
      else setTwilioMsgSending(false);
    }
  };

  if (!isEmergency) {
    return (
      <div style={{ padding: '14px', background: 'rgba(16, 185, 129, 0.08)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(16, 185, 129, 0.2)', fontSize: '13px', color: '#34d399', textAlign: 'center' }}>
        ✅ Scene classified as <strong>Normal</strong>. Emergency notification dispatchers are in standby.
      </div>
    );
  }

  return (
    <div className="emergency-actions-card" style={{ background: 'rgba(255, 90, 95, 0.08)', border: '1px solid rgba(255, 90, 95, 0.25)', borderRadius: 'var(--radius-md)', padding: '20px', marginTop: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
        <div>
          <div style={{ fontSize: '14.5px', fontWeight: '700', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>🚨</span>
            <span>Emergency Response Actions</span>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Trigger real-time notifications for confirmed {prediction} event (Confidence: {(confidence * 100).toFixed(2)}%)
          </div>
        </div>
        <span className="badge badge-fire">Active Event</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
        {/* WhatsApp Button */}
        <div>
          <button
            className="btn btn-warning"
            style={{ width: '100%' }}
            onClick={handleWhatsApp}
            disabled={waSending || waStatus === 'success'}
            id="btn-whatsapp-alert"
          >
            {waSending ? 'Opening WhatsApp...' : '📱 Send WhatsApp Alert'}
          </button>
          {waStatus === 'success' && <div style={{ color: '#10B981', fontSize: '11.5px', marginTop: '4px', textAlign: 'center', fontWeight: '600' }}>✓ WhatsApp Link Opened</div>}
          {waStatus === 'error' && <div style={{ color: '#FF5A5F', fontSize: '11.5px', marginTop: '4px', textAlign: 'center', fontWeight: '600' }}>✕ Dispatch Failed</div>}
        </div>

        {/* Twilio Call Button */}
        <div>
          <button
            className="btn btn-danger"
            style={{ width: '100%' }}
            onClick={() => handleTwilio('call')}
            disabled={twilioCallSending || twilioCallStatus === 'success'}
            id="btn-twilio-call"
          >
            {twilioCallSending ? 'Calling...' : `📞 Call ${prediction === 'Fire' ? 'Fire Dept' : 'Ambulance'}`}
          </button>
          {twilioCallStatus === 'success' && <div style={{ color: '#10B981', fontSize: '11.5px', marginTop: '4px', textAlign: 'center', fontWeight: '600' }}>✓ Call Initiated</div>}
          {twilioCallStatus === 'error' && <div style={{ color: '#FF5A5F', fontSize: '11.5px', marginTop: '4px', textAlign: 'center', fontWeight: '600' }}>✕ Call Failed</div>}
        </div>

        {/* Twilio SMS Button */}
        <div>
          <button
            className="btn btn-accent"
            style={{ width: '100%' }}
            onClick={() => handleTwilio('message')}
            disabled={twilioMsgSending || twilioMsgStatus === 'success'}
            id="btn-twilio-message"
          >
            {twilioMsgSending ? 'Sending SMS...' : `💬 Send ${prediction} SMS`}
          </button>
          {twilioMsgStatus === 'success' && <div style={{ color: '#10B981', fontSize: '11.5px', marginTop: '4px', textAlign: 'center', fontWeight: '600' }}>✓ SMS Dispatched</div>}
          {twilioMsgStatus === 'error' && <div style={{ color: '#FF5A5F', fontSize: '11.5px', marginTop: '4px', textAlign: 'center', fontWeight: '600' }}>✕ SMS Failed</div>}
        </div>
      </div>
    </div>
  );
}
