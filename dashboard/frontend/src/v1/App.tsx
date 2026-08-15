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
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  Signal,
  Target,
} from 'lucide-react';
import './index.css';

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

type RoomSectionHandle = {
  play: () => Promise<void>;
  pause: () => void;
  reset: () => void;
};

type RoomSectionProps = {
  title: string;
  role: string;
  videoName: string;
  actualLabel: string;
};

const formatLabel = (label: string) => label.replace('_', ' ').toUpperCase();

const RoomSection = forwardRef<RoomSectionHandle, RoomSectionProps>(
  ({ title, role, videoName, actualLabel }, ref) => {
    const [data, setData] = useState<CsiData | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');
    const [currentTime, setCurrentTime] = useState<number>(START_TIME);

    const videoRef = useRef<HTMLVideoElement>(null);

    const fetchData = useCallback(() => {
      setLoading(true);
      setError('');
      const csvName = videoName.replace('.mp4', '.csv');
      axios
        .get<CsiData>(`${API_BASE}/api/data/${csvName}`)
        .then((res) => {
          setData(res.data);
          setLoading(false);
        })
        .catch((err: Error) => {
          setError(err.message || 'Failed to fetch data');
          setLoading(false);
          setData(null);
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
      if (!videoRef.current) return;

      const video = videoRef.current;
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
      const idx = Math.floor((currentTime - START_TIME) * PACKETS_PER_SECOND);
      return Math.max(0, Math.min(idx, data.total_packets - 1));
    }, [currentTime, data]);

    const currentAmplitudeData = useMemo(() => {
      if (!data || !data.magnitude[currentPacketIndex]) return [];
      return data.magnitude[currentPacketIndex].map((amp, idx) => ({
        subcarrier: idx,
        amplitude: amp,
      }));
    }, [data, currentPacketIndex]);

    const currentPrediction = useMemo(() => {
      if (!data || data.predictions.length === 0) {
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

    const motionTimelineData = useMemo(() => {
      if (!data) return [];
      const startIdx = Math.max(0, currentPacketIndex - 400);
      const slice = data.motion_metric.slice(startIdx, currentPacketIndex + 1);
      return slice.map((val, idx) => ({
        time: ((startIdx + idx) / PACKETS_PER_SECOND).toFixed(1),
        motion: val,
      }));
    }, [data, currentPacketIndex]);

    const isWaiting = currentPrediction.prediction === 'WAITING...';
    const isCorrect = currentPrediction.prediction === actualLabel;

    return (
      <article className="comparison-panel">
        <div className="comparison-header">
          <div>
            <p className="eyebrow">{role}</p>
            <h2>{title}</h2>
          </div>
          <span className="packet-chip">PKT {currentPacketIndex}</span>
        </div>

        {error && (
          <div className="error-banner">
            <AlertTriangle size={16} />
            <span>CSI feed unavailable: {error}</span>
            <button onClick={fetchData} className="icon-btn" aria-label={`Retry ${title}`}>
              <RefreshCw size={15} />
            </button>
          </div>
        )}

        <div className="evidence-grid">
          <div className="video-stack">
            <div className="video-frame">
              <video
                ref={videoRef}
                src={`${API_BASE}/media/${videoName}`}
                muted
                playsInline
                onLoadedMetadata={handleLoadedMetadata}
                onTimeUpdate={handleTimeUpdate}
              />
              <span className="video-label">{videoName}</span>
            </div>
            <div className="status-strip">
              <div>
                <span>Ground truth</span>
                <strong>{formatLabel(actualLabel)}</strong>
              </div>
              <div className={isWaiting ? 'waiting' : isCorrect ? 'confirmed' : 'flagged'}>
                <span>Model read</span>
                <strong>{formatLabel(currentPrediction.prediction)}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>
                  {isWaiting ? '--' : `${(currentPrediction.confidence * 100).toFixed(1)}%`}
                </strong>
              </div>
            </div>
          </div>

          <div className="signal-stack">
            <div className="mini-panel">
              <div className="panel-heading">
                <Signal size={15} />
                <span>CSI amplitude</span>
              </div>
              <div className="chart-container">
                {loading && <div className="loading-overlay">Syncing feed</div>}
                {!loading && currentAmplitudeData.length > 0 && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentAmplitudeData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(213, 218, 211, 0.08)" />
                      <XAxis dataKey="subcarrier" stroke="rgba(213, 218, 211, 0.45)" tick={{ fontSize: 10 }} />
                      <YAxis stroke="rgba(213, 218, 211, 0.45)" tick={{ fontSize: 10 }} domain={['auto', 'auto']} width={32} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(11, 13, 12, 0.94)',
                          border: '1px solid rgba(188, 198, 181, 0.2)',
                          borderRadius: '4px',
                          color: '#eef2ea',
                        }}
                      />
                      <Line type="monotone" dataKey="amplitude" stroke="#a8b88f" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
                {!loading && !error && currentAmplitudeData.length === 0 && (
                  <div className="empty-state">Awaiting packets</div>
                )}
              </div>
            </div>

            <div className="mini-panel">
              <div className="panel-heading">
                <Activity size={15} />
                <span>Motion energy</span>
              </div>
              <div className="chart-container">
                {loading && <div className="loading-overlay">Syncing feed</div>}
                {!loading && motionTimelineData.length > 0 && (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={motionTimelineData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(213, 218, 211, 0.08)" />
                      <XAxis dataKey="time" stroke="rgba(213, 218, 211, 0.45)" tick={{ fontSize: 10 }} />
                      <YAxis stroke="rgba(213, 218, 211, 0.45)" tick={{ fontSize: 10 }} domain={[0, 'auto']} width={32} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(11, 13, 12, 0.94)',
                          border: '1px solid rgba(188, 198, 181, 0.2)',
                          borderRadius: '4px',
                          color: '#eef2ea',
                        }}
                      />
                      <Line type="monotone" dataKey="motion" stroke="#d5d9c6" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
                {!loading && !error && motionTimelineData.length === 0 && (
                  <div className="empty-state">Awaiting motion</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </article>
    );
  },
);

function App() {
  const emptyRoomRef = useRef<RoomSectionHandle>(null);
  const occupiedRoomRef = useRef<RoomSectionHandle>(null);
  const [playing, setPlaying] = useState(false);

  const handleTogglePlayback = async () => {
    if (playing) {
      emptyRoomRef.current?.pause();
      occupiedRoomRef.current?.pause();
      setPlaying(false);
      return;
    }

    try {
      await Promise.all([
        emptyRoomRef.current?.play(),
        occupiedRoomRef.current?.play(),
      ]);
      setPlaying(true);
    } catch {
      setPlaying(false);
    }
  };

  const handleReset = () => {
    emptyRoomRef.current?.reset();
    occupiedRoomRef.current?.reset();
    setPlaying(false);
  };

  return (
    <div className="dashboard-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <Radio size={24} />
          <div>
            <p className="eyebrow">WiTrack field demonstrator</p>
            <h1>Wi-Fi occupancy analysis</h1>
          </div>
        </div>
        <div className="control-cluster">
          <button className="run-button" onClick={handleTogglePlayback}>
            {playing ? <Pause size={18} /> : <Play size={18} />}
            <span>{playing ? 'Pause both' : 'Run both'}</span>
          </button>
          <button className="icon-btn reset-btn" onClick={handleReset} aria-label="Reset demo">
            <RotateCcw size={17} />
          </button>
        </div>
        <div className="mission-summary">
          <div>
            <Target size={15} />
            <span>2 rooms</span>
          </div>
          <div>
            <CheckCircle2 size={15} />
            <span>CSI + video sync</span>
          </div>
        </div>
      </header>

      <main className="comparison-grid">
        <RoomSection
          ref={emptyRoomRef}
          title="Bathroom clear"
          role="control lane"
          videoName="bathroom_empty.mp4"
          actualLabel="no_activity"
        />
        <RoomSection
          ref={occupiedRoomRef}
          title="Bathroom occupied"
          role="activity lane"
          videoName="bathroom_group.mp4"
          actualLabel="activity"
        />
      </main>
    </div>
  );
}

export default App;
