import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import axios from 'axios';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AlertTriangle, Pause, Play, RefreshCw, RotateCcw } from 'lucide-react';

const API_BASE = 'http://localhost:8000';
const START_TIME = 30;
const PACKETS_PER_SECOND = 100;

type Prediction = {
  index: number;
  prediction: string;
  confidence: number;
};

type CsiData = {
  magnitude: number[][];
  motion_metric: number[];
  predictions: Prediction[];
  total_packets: number;
};

type RoomChannelHandle = {
  play: () => Promise<void>;
  pause: () => void;
  reset: () => void;
};

type RoomChannelProps = {
  callSign: string;
  title: string;
  videoName: string;
  actualLabel: string;
};

const formatLabel = (label: string) => label.replace('_', ' ').toUpperCase();

const RoomChannel = forwardRef<RoomChannelHandle, RoomChannelProps>(
  ({ callSign, title, videoName, actualLabel }, ref) => {
    const [data, setData] = useState<CsiData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [currentTime, setCurrentTime] = useState(START_TIME);

    const videoRef = useRef<HTMLVideoElement>(null);

    const fetchData = useCallback(() => {
      setLoading(true);
      setError('');

      axios
        .get<CsiData>(`${API_BASE}/api/data/${videoName.replace('.mp4', '.csv')}`)
        .then((res) => {
          setData(res.data);
          setLoading(false);
        })
        .catch((err: Error) => {
          setError(err.message || 'Failed to fetch CSI data');
          setData(null);
          setLoading(false);
        });
    }, [videoName]);

    useEffect(() => {
      fetchData();
    }, [fetchData]);

    useImperativeHandle(ref, () => ({
      play: async () => {
        const video = videoRef.current;
        if (!video) return;
        if (video.currentTime < START_TIME || video.ended) {
          video.currentTime = START_TIME;
        }
        await video.play();
      },
      pause: () => {
        videoRef.current?.pause();
      },
      reset: () => {
        const video = videoRef.current;
        if (!video) return;
        video.pause();
        video.currentTime = START_TIME;
        setCurrentTime(START_TIME);
      },
    }));

    const handleLoadedMetadata = () => {
      if (videoRef.current) {
        videoRef.current.currentTime = START_TIME;
      }
    };

    const handleTimeUpdate = () => {
      const video = videoRef.current;
      if (!video) return;

      if (video.currentTime < START_TIME) {
        video.currentTime = START_TIME;
      }

      if (video.duration && video.currentTime >= video.duration - START_TIME) {
        video.pause();
      }

      setCurrentTime(video.currentTime);
    };

    const currentPacketIndex = useMemo(() => {
      if (!data) return 0;
      const packet = Math.floor((currentTime - START_TIME) * PACKETS_PER_SECOND);
      return Math.max(0, Math.min(packet, data.total_packets - 1));
    }, [currentTime, data]);

    const amplitudeData = useMemo(() => {
      if (!data?.magnitude[currentPacketIndex]) return [];
      return data.magnitude[currentPacketIndex].map((amplitude, subcarrier) => ({
        amplitude,
        subcarrier,
      }));
    }, [data, currentPacketIndex]);

    const motionData = useMemo(() => {
      if (!data) return [];
      const startIdx = Math.max(0, currentPacketIndex - 400);
      return data.motion_metric.slice(startIdx, currentPacketIndex + 1).map((motion, idx) => ({
        motion,
        time: ((startIdx + idx) / PACKETS_PER_SECOND).toFixed(1),
      }));
    }, [data, currentPacketIndex]);

    const currentPrediction = useMemo(() => {
      if (!data?.predictions.length) {
        return { prediction: 'WAITING...', confidence: 0 };
      }

      let latest = data.predictions[0];
      for (let i = 0; i < data.predictions.length; i += 1) {
        if (data.predictions[i].index <= currentPacketIndex) {
          latest = data.predictions[i];
        } else {
          break;
        }
      }
      return latest;
    }, [data, currentPacketIndex]);

    const isWaiting = currentPrediction.prediction === 'WAITING...';
    const isCorrect = currentPrediction.prediction === actualLabel;
    const confidence = isWaiting ? '--' : `${(currentPrediction.confidence * 100).toFixed(1)}%`;

    return (
      <section className="channel" aria-label={title}>
        <div className="channel-title">
          <div>
            <span>{callSign}</span>
            <h2>{title}</h2>
          </div>
          <b>PKT {currentPacketIndex}</b>
        </div>

        <div className="channel-glass">
          <div className="crt-window">
            <video
              ref={videoRef}
              src={`${API_BASE}/media/${videoName}`}
              muted
              playsInline
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
            />
            <div className="screen-vignette" />
            <div className="screen-scanline" />
            <span className="source-tag">{videoName}</span>
          </div>

          <div className="telemetry">
            <div className="chart-block">
              <div className="chart-title">CSI amplitude</div>
              <div className="chart-area">
                {loading && <div className="loading-state">Acquiring packets</div>}
                {!loading && amplitudeData.length > 0 && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={amplitudeData} margin={{ top: 6, right: 10, bottom: 24, left: 12 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke="rgba(231, 234, 220, 0.08)" />
                      <XAxis
                        dataKey="subcarrier"
                        stroke="rgba(231, 234, 220, 0.48)"
                        tick={{ fontSize: 9 }}
                        label={{
                          value: 'Subcarrier index',
                          position: 'insideBottom',
                          offset: -14,
                          fill: 'rgba(231, 234, 220, 0.7)',
                          fontSize: 10,
                        }}
                      />
                      <YAxis
                        stroke="rgba(231, 234, 220, 0.48)"
                        tick={{ fontSize: 9 }}
                        width={32}
                        domain={['auto', 'auto']}
                        label={{
                          value: 'Amplitude',
                          angle: -90,
                          position: 'insideLeft',
                          fill: 'rgba(231, 234, 220, 0.7)',
                          fontSize: 10,
                        }}
                      />
                      <Tooltip contentStyle={{ background: '#10120f', border: '1px solid rgba(219, 222, 208, 0.26)', borderRadius: 3, color: '#f3f0db' }} />
                      <Line type="monotone" dataKey="amplitude" stroke="#f0d47a" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
                {!loading && !error && amplitudeData.length === 0 && (
                  <div className="loading-state">No amplitude packets</div>
                )}
              </div>
            </div>

            <div className="chart-block">
              <div className="chart-title">Motion energy</div>
              <div className="chart-area">
                {loading && <div className="loading-state">Acquiring packets</div>}
                {!loading && motionData.length > 0 && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={motionData} margin={{ top: 6, right: 10, bottom: 24, left: 12 }}>
                      <CartesianGrid strokeDasharray="2 4" stroke="rgba(231, 234, 220, 0.08)" />
                      <XAxis
                        dataKey="time"
                        stroke="rgba(231, 234, 220, 0.48)"
                        tick={{ fontSize: 9 }}
                        label={{
                          value: 'Window time (s)',
                          position: 'insideBottom',
                          offset: -14,
                          fill: 'rgba(231, 234, 220, 0.7)',
                          fontSize: 10,
                        }}
                      />
                      <YAxis
                        stroke="rgba(231, 234, 220, 0.48)"
                        tick={{ fontSize: 9 }}
                        width={32}
                        domain={[0, 'auto']}
                        label={{
                          value: 'Energy',
                          angle: -90,
                          position: 'insideLeft',
                          fill: 'rgba(231, 234, 220, 0.7)',
                          fontSize: 10,
                        }}
                      />
                      <Tooltip contentStyle={{ background: '#10120f', border: '1px solid rgba(219, 222, 208, 0.26)', borderRadius: 3, color: '#f3f0db' }} />
                      <Line type="monotone" dataKey="motion" stroke="#c8d6b4" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
                {!loading && !error && motionData.length === 0 && (
                  <div className="loading-state">No motion window</div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="status-row">
          <div className={`stat-crt ${isWaiting ? 'pending' : isCorrect ? 'pass' : 'fail'}`}>
            <span>Ground truth</span>
            <strong>{formatLabel(actualLabel)}</strong>
          </div>
          <div className={`stat-crt ${isWaiting ? 'pending' : isCorrect ? 'pass' : 'fail'}`}>
            <span>Model read</span>
            <strong>{formatLabel(currentPrediction.prediction)}</strong>
          </div>
          <div className={`stat-crt ${isWaiting ? 'pending' : isCorrect ? 'pass' : 'fail'}`}>
            <span>Confidence</span>
            <strong>{confidence}</strong>
          </div>
          {error && (
            <button className="retry-link" onClick={fetchData}>
              <AlertTriangle size={13} />
              <span>Retry feed</span>
              <RefreshCw size={13} />
            </button>
          )}
        </div>
      </section>
    );
  },
);

function App() {
  const clearRoomRef = useRef<RoomChannelHandle>(null);
  const activeRoomRef = useRef<RoomChannelHandle>(null);
  const [playing, setPlaying] = useState(false);

  const handleTogglePlayback = async () => {
    if (playing) {
      clearRoomRef.current?.pause();
      activeRoomRef.current?.pause();
      setPlaying(false);
      return;
    }

    try {
      await Promise.all([clearRoomRef.current?.play(), activeRoomRef.current?.play()]);
      setPlaying(true);
    } catch {
      setPlaying(false);
    }
  };

  const handleReset = () => {
    clearRoomRef.current?.reset();
    activeRoomRef.current?.reset();
    setPlaying(false);
  };

  return (
    <main className="demo-stage">
      <header className="steel-heading">
        <div>
          <p>Field-grade wireless sensing</p>
          <h1>WiTrack</h1>
        </div>
        <div className="primary-controls">
          <button className="control-btn" onClick={handleTogglePlayback}>
            {playing ? <Pause size={18} /> : <Play size={18} />}
            <span>{playing ? 'Pause both' : 'Run both'}</span>
          </button>
          <button className="control-btn-icon" onClick={handleReset} aria-label="Reset synchronized demo">
            <RotateCcw size={17} />
          </button>
        </div>
      </header>

      <div className="tv-body">
        <div className="cabinet-lip" />
        <div className="screen-deck">
          <RoomChannel
            ref={clearRoomRef}
            callSign="Control channel"
            title="Bathroom clear"
            videoName="bathroom_empty.mp4"
            actualLabel="no_activity"
          />
          <RoomChannel
            ref={activeRoomRef}
            callSign="Activity channel"
            title="Bathroom occupied"
            videoName="bathroom_group.mp4"
            actualLabel="activity"
          />
        </div>
        <div className="console-controls" aria-hidden="true">
          <div className="screw" />
          <div className="vent-bank">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
          <div className="brass-plate">SYNCHRONIZED CSI VALIDATION</div>
          <div className="vent-bank">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
          <div className="screw angled" />
        </div>
      </div>
    </main>
  );
}

export default App;
