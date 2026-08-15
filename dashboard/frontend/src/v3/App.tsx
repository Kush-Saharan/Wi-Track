import { useEffect, useRef, useState, useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import './index.css';

const WS_URL = `ws://${window.location.hostname}:8000/ws/data`;

type RealtimePayload = {
  amplitude: number[];
  prediction: string;
  confidence: number;
  motion: number;
  timestamp: number;
};

const formatLabel = (label: string) => label.replace('_', ' ').toUpperCase();

function App() {
  const [data, setData] = useState<RealtimePayload | null>(null);
  const [motionHistory, setMotionHistory] = useState<{ time: string, motion: number }[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const packetCountRef = useRef(0);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err: any) {
      console.error("Camera error:", err);
      setError("Failed to access camera: " + err.message);
    }
  };

  const connectWebSocket = () => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
      setConnected(true);
      setError('');
    };
    
    ws.onmessage = (event) => {
      try {
        const payload: RealtimePayload = JSON.parse(event.data);
        packetCountRef.current += 5; // 5 packets per batch
        setData(payload);
        setMotionHistory(prev => {
          const newH = [...prev, { time: (packetCountRef.current / 100).toFixed(1), motion: payload.motion }];
          if (newH.length > 100) return newH.slice(-100);
          return newH;
        });
      } catch (e) {
        console.error("Error parsing WS data", e);
      }
    };
    
    ws.onerror = (e) => {
      setConnected(false);
      setError("WebSocket connection failed");
    };
    
    ws.onclose = () => {
      setConnected(false);
    };
    
    wsRef.current = ws;
  };

  useEffect(() => {
    startCamera();
    connectWebSocket();
    return () => {
      wsRef.current?.close();
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  const amplitudeData = useMemo(() => {
    if (!data) return [];
    return data.amplitude.map((amp, idx) => ({ amplitude: amp, subcarrier: idx }));
  }, [data]);

  const currentPrediction = data?.prediction || 'WAITING...';
  // Use a generic logic to decide pass/fail based on active or empty.
  // Since we don't have a specific ground truth, we just display the prediction
  // and light it up if it detects activity.
  const isWaiting = currentPrediction === 'WAITING...';
  const hasActivity = currentPrediction.toLowerCase().includes('activity') && currentPrediction !== 'no_activity';
  const isNoActivity = currentPrediction === 'no_activity';
  
  const statusClass = isWaiting ? 'pending' : (hasActivity ? 'fail' : 'pass'); // Red for activity, Green for no activity

  return (
    <main className="demo-stage" style={{ gridTemplateRows: 'auto 1fr' }}>
      <header className="steel-heading">
        <div>
          <p>Field-grade wireless sensing</p>
          <h1>WiTrack Live</h1>
        </div>
        <div className="primary-controls">
          <button className="control-btn" onClick={connectWebSocket}>
            <RefreshCw size={18} />
            <span>Reconnect</span>
          </button>
        </div>
      </header>

      <div className="tv-body">
        <div className="cabinet-lip" />
        <div className="screen-deck" style={{ gridTemplateColumns: '1fr' }}>
          
          <section className="channel" aria-label="Real-time Stream">
            <div className="channel-title">
              <div>
                <span>LIVE FEED</span>
                <h2>Real-Time Monitor</h2>
              </div>
              <b>PKT {packetCountRef.current}</b>
            </div>

            <div className="channel-glass">
              <div className="crt-window">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                />
                <div className="screen-vignette" />
                <div className="screen-scanline" />
                <span className="source-tag">LOCAL_WEBCAM_0</span>
              </div>

              <div className="telemetry">
                <div className="chart-block">
                  <div className="chart-title">Live CSI Amplitude</div>
                  <div className="chart-area">
                    {!connected && <div className="loading-state">Connecting to ESP32...</div>}
                    {connected && amplitudeData.length > 0 && (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={amplitudeData} margin={{ top: 6, right: 10, bottom: 24, left: 12 }}>
                          <CartesianGrid strokeDasharray="2 4" stroke="rgba(231, 234, 220, 0.08)" />
                          <XAxis dataKey="subcarrier" stroke="rgba(231, 234, 220, 0.48)" tick={{ fontSize: 9 }} />
                          <YAxis stroke="rgba(231, 234, 220, 0.48)" tick={{ fontSize: 9 }} width={32} domain={['auto', 'auto']} />
                          <Tooltip contentStyle={{ background: '#10120f', border: '1px solid rgba(219, 222, 208, 0.26)', borderRadius: 3, color: '#f3f0db' }} />
                          <Line type="monotone" dataKey="amplitude" stroke="#f0d47a" strokeWidth={2} dot={false} isAnimationActive={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>

                <div className="chart-block">
                  <div className="chart-title">Motion Energy Trend</div>
                  <div className="chart-area">
                    {!connected && <div className="loading-state">Connecting to ESP32...</div>}
                    {connected && motionHistory.length > 0 && (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={motionHistory} margin={{ top: 6, right: 10, bottom: 24, left: 12 }}>
                          <CartesianGrid strokeDasharray="2 4" stroke="rgba(231, 234, 220, 0.08)" />
                          <XAxis dataKey="time" stroke="rgba(231, 234, 220, 0.48)" tick={{ fontSize: 9 }} />
                          <YAxis stroke="rgba(231, 234, 220, 0.48)" tick={{ fontSize: 9 }} width={32} domain={[0, 'auto']} />
                          <Tooltip contentStyle={{ background: '#10120f', border: '1px solid rgba(219, 222, 208, 0.26)', borderRadius: 3, color: '#f3f0db' }} />
                          <Line type="monotone" dataKey="motion" stroke="#c8d6b4" strokeWidth={2} dot={false} isAnimationActive={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="status-row">
              <div className={`stat-crt ${statusClass}`}>
                <span>System Status</span>
                <strong>{connected ? 'ONLINE' : 'OFFLINE'}</strong>
              </div>
              <div className={`stat-crt ${statusClass}`}>
                <span>Live Model Read</span>
                <strong>{formatLabel(currentPrediction)}</strong>
              </div>
              <div className={`stat-crt ${statusClass}`}>
                <span>Confidence</span>
                <strong>{isWaiting ? '--' : `${(data.confidence * 100).toFixed(1)}%`}</strong>
              </div>
            </div>
            
            {error && (
              <div style={{ color: '#ec6b5f', textAlign: 'center', marginTop: '10px', fontSize: '14px' }}>
                <AlertTriangle size={14} style={{ display: 'inline', marginRight: 4 }} /> {error}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

export default App;
