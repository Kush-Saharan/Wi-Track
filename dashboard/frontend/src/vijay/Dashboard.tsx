import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import "./vijay.css";
import { CSIWebSocket } from "./websocket";
import type {
  CSIData,
  YOLOData,
  ModelData,
} from "./websocket";

const WS_URL = "ws://localhost:8000/ws";

const MAX_GRAPH_POINTS = 192;

interface RecordingSession {
  sessionId: string;
  startedAt: string;
  stoppedAt: string;
  duration: number;
  samples: CSIData[];
}

/*
 * Frontend view of a YOLO detection.
 *
 * The backend can later send x/y/width/height
 * as normalized 0..1 values or pixel values.
 * The camera overlay converts either form
 * into percentages.
 */
interface YOLODetectionView {
  label: string;
  confidence: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

function formatTime(seconds: number) {
  const safeSeconds = Math.max(
    0,
    Math.floor(seconds)
  );

  const hours = Math.floor(
    safeSeconds / 3600
  );

  const minutes = Math.floor(
    (safeSeconds % 3600) / 60
  );

  const secs = safeSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(
      2,
      "0"
    )}:${String(minutes).padStart(
      2,
      "0"
    )}:${String(secs).padStart(
      2,
      "0"
    )}`;
  }

  return `${String(minutes).padStart(
    2,
    "0"
  )}:${String(secs).padStart(
    2,
    "0"
  )}`;
}

function createSessionId() {
  const now = new Date();

  const date = now
    .toISOString()
    .slice(0, 10)
    .replace(/-/g, "");

  const time = now
    .toTimeString()
    .slice(0, 8)
    .replace(/:/g, "");

  return `WT-${date}-${time}`;
}

function getTimelineRange(seconds: number) {
  const minimumRange = 60;

  if (seconds <= minimumRange) {
    return minimumRange;
  }

  return Math.ceil(
    seconds / 60
  ) * 60;
}

function getTimelineStep(range: number) {
  if (range <= 60) {
    return 10;
  }

  if (range <= 120) {
    return 20;
  }

  if (range <= 300) {
    return 60;
  }

  if (range <= 600) {
    return 120;
  }

  return 300;
}

function toPercent(
  value: number | undefined,
  total: number | undefined
) {
  if (value === undefined) {
    return undefined;
  }

  /*
   * Support both normalized YOLO coordinates
   * (0..1) and pixel coordinates.
   */
  if (
    value >= 0 &&
    value <= 1
  ) {
    return value * 100;
  }

  if (
    total &&
    total > 0
  ) {
    return (
      (value / total) * 100
    );
  }

  return undefined;
}


interface DownloadEntry {
  name: string;
  data: Uint8Array;
}

const textEncoder = new TextEncoder();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;

  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];

    for (let bit = 0; bit < 8; bit++) {
      crc =
        (crc >>> 1) ^
        (crc & 1
          ? 0xedb88320
          : 0);
    }
  }

  return (crc ^ 0xffffffff) >>> 0;
}

function writeUint16(
  view: DataView,
  offset: number,
  value: number
) {
  view.setUint16(
    offset,
    value,
    true
  );
}

function writeUint32(
  view: DataView,
  offset: number,
  value: number
) {
  view.setUint32(
    offset,
    value >>> 0,
    true
  );
}

/*
 * Creates a ZIP file using the ZIP "store"
 * method. No extra package is required.
 *
 * This lets us put CSI, video, YOLO and MODEL
 * data into one downloadable file.
 */
function createZip(
  entries: DownloadEntry[]
): Blob {
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];

  let offset = 0;

  for (const entry of entries) {
    const nameBytes =
      textEncoder.encode(
        entry.name
      );

    const data = entry.data;

    const checksum =
      crc32(data);

    const localHeader =
      new ArrayBuffer(30);

    const localView =
      new DataView(
        localHeader
      );

    writeUint32(
      localView,
      0,
      0x04034b50
    );

    writeUint16(
      localView,
      4,
      20
    );

    writeUint16(
      localView,
      6,
      0x0800
    );

    writeUint16(
      localView,
      8,
      0
    );

    writeUint16(
      localView,
      10,
      0
    );

    writeUint16(
      localView,
      12,
      0
    );

    writeUint32(
      localView,
      14,
      checksum
    );

    writeUint32(
      localView,
      18,
      data.length
    );

    writeUint32(
      localView,
      22,
      data.length
    );

    writeUint16(
      localView,
      26,
      nameBytes.length
    );

    writeUint16(
      localView,
      28,
      0
    );

    localParts.push(
      new Uint8Array(
        localHeader
      ),
      nameBytes,
      data
    );

    const centralHeader =
      new ArrayBuffer(46);

    const centralView =
      new DataView(
        centralHeader
      );

    writeUint32(
      centralView,
      0,
      0x02014b50
    );

    writeUint16(
      centralView,
      4,
      20
    );

    writeUint16(
      centralView,
      6,
      20
    );

    writeUint16(
      centralView,
      8,
      0x0800
    );

    writeUint16(
      centralView,
      10,
      0
    );

    writeUint16(
      centralView,
      12,
      0
    );

    writeUint16(
      centralView,
      14,
      0
    );

    writeUint32(
      centralView,
      16,
      checksum
    );

    writeUint32(
      centralView,
      20,
      data.length
    );

    writeUint32(
      centralView,
      24,
      data.length
    );

    writeUint16(
      centralView,
      28,
      nameBytes.length
    );

    writeUint16(
      centralView,
      30,
      0
    );

    writeUint16(
      centralView,
      32,
      0
    );

    writeUint16(
      centralView,
      34,
      0
    );

    writeUint16(
      centralView,
      36,
      0
    );

    writeUint32(
      centralView,
      38,
      0
    );

    writeUint32(
      centralView,
      42,
      offset
    );

    centralParts.push(
      new Uint8Array(
        centralHeader
      ),
      nameBytes
    );

    offset +=
      30 +
      nameBytes.length +
      data.length;
  }

  const centralDirectorySize =
    centralParts.reduce(
      (total, part) =>
        total + part.length,
      0
    );

  const centralDirectoryOffset =
    offset;

  const endRecord =
    new ArrayBuffer(22);

  const endView =
    new DataView(
      endRecord
    );

  writeUint32(
    endView,
    0,
    0x06054b50
  );

  writeUint16(
    endView,
    4,
    0
  );

  writeUint16(
    endView,
    6,
    0
  );

  writeUint16(
    endView,
    8,
    entries.length
  );

  writeUint16(
    endView,
    10,
    entries.length
  );

  writeUint32(
    endView,
    12,
    centralDirectorySize
  );

  writeUint32(
    endView,
    16,
    centralDirectoryOffset
  );

  writeUint16(
    endView,
    20,
    0
  );

  return new Blob(
    [
      ...localParts,
      ...centralParts,
      new Uint8Array(
        endRecord
      ),
    ],
    {
      type:
        "application/zip",
    }
  );
}

function blobToUint8Array(
  blob: Blob
): Promise<Uint8Array> {
  return blob
    .arrayBuffer()
    .then(
      (buffer) =>
        new Uint8Array(buffer)
    );
}

function Dashboard() {
  const [isRecording, setIsRecording] =
    useState(false);

  /*
   * Actual recording time.
   */
  const [elapsedTime, setElapsedTime] =
    useState(0);

  /*
   * Time currently selected by
   * the inspection slider.
   */
  const [selectedTime, setSelectedTime] =
    useState(0);

  const [sessionId, setSessionId] =
    useState<string | null>(null);

  const [startedAt, setStartedAt] =
    useState<string | null>(null);

  const [lastSession, setLastSession] =
    useState<RecordingSession | null>(
      null
    );

  const [connectionStatus, setConnectionStatus] =
    useState<
      | "DISCONNECTED"
      | "CONNECTING"
      | "CONNECTED"
      | "ERROR"
    >("DISCONNECTED");

  const [cameraStatus, setCameraStatus] =
    useState<
      | "OFF"
      | "STARTING"
      | "LIVE"
      | "ERROR"
    >("OFF");

  const [cameraName, setCameraName] =
    useState("No camera selected");

  const [videoStatus, setVideoStatus] =
    useState<
      | "IDLE"
      | "RECORDING"
      | "PROCESSING"
      | "READY"
    >("IDLE");

  /*
   * Current I/Q values received from the
   * backend WebSocket.
   */
  const [iData, setIData] =
    useState<number[]>([]);

  const [qData, setQData] =
    useState<number[]>([]);

  const [sampleCount, setSampleCount] =
    useState(0);

  const recordedSamples =
    useRef<CSIData[]>([]);

  const [yoloHistory, setYoloHistory] =
    useState<YOLOData[]>([]);

  const [currentYOLO, setCurrentYOLO] =
    useState<YOLOData | null>(null);

  const [modelHistory, setModelHistory] =
    useState<ModelData[]>([]);

  const [currentModel, setCurrentModel] =
    useState<ModelData | null>(null);

  const websocketRef =
    useRef<CSIWebSocket | null>(null);

  const videoRef =
    useRef<HTMLVideoElement | null>(null);

  const cameraStreamRef =
    useRef<MediaStream | null>(null);

  const videoChunksRef =
    useRef<Blob[]>([]);

  const mediaRecorderRef =
    useRef<MediaRecorder | null>(null);

  const recordedVideoRef =
    useRef<Blob | null>(null);

  const videoPlaybackUrl =
    useRef<string | null>(null);

  /*
   * REAL-TIME recording timer.
   */
  useEffect(() => {
    if (!isRecording) {
      return;
    }

    const timer =
      window.setInterval(() => {
        setElapsedTime(
          (previous) =>
            previous + 1
        );

        /*
         * During live recording,
         * inspection follows NOW.
         */
        setSelectedTime(
          (previous) =>
            previous + 1
        );
      }, 1000);

    return () => {
      window.clearInterval(
        timer
      );
    };
  }, [isRecording]);

  /*
   * Cleanup.
   */
  useEffect(() => {
    return () => {
      websocketRef.current?.disconnect();

      cameraStreamRef.current
        ?.getTracks()
        .forEach((track) => {
          track.stop();
        });

      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current
          .state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }

      if (
        videoPlaybackUrl.current
      ) {
        URL.revokeObjectURL(
          videoPlaybackUrl.current
        );
      }
    };
  }, []);

  /*
   * Camera preview.
   */
  useEffect(() => {
    if (
      cameraStatus === "LIVE" &&
      videoRef.current &&
      cameraStreamRef.current
    ) {
      videoRef.current.srcObject =
        cameraStreamRef.current;

      videoRef.current
        .play()
        .catch(() => {});
    }
  }, [cameraStatus]);

  /*
   * Seek recorded video when
   * selectedTime changes.
   */
  useEffect(() => {
    if (
      isRecording ||
      !videoPlaybackUrl.current ||
      !videoRef.current
    ) {
      return;
    }

    const video =
      videoRef.current;

    video.srcObject = null;

    video.src =
      videoPlaybackUrl.current;

    video.currentTime =
      selectedTime;

    video.load();
  }, [
    isRecording,
    selectedTime,
  ]);

  /*
   * Find the backend CSI frame nearest to
   * the selected inspection timestamp.
   */
  const showCSIAtTime = (time: number) => {
    if (
      recordedSamples.current.length === 0 ||
      !startedAt
    ) {
      return;
    }

    const startTimestamp =
      new Date(startedAt).getTime();

    let closest =
      recordedSamples.current[0];

    let closestDifference =
      Infinity;

    for (const sample of recordedSamples.current) {
      const sampleTime =
        (sample.timestamp - startTimestamp) /
        1000;

      const difference =
        Math.abs(sampleTime - time);

      if (difference < closestDifference) {
        closest = sample;
        closestDifference = difference;
      }
    }

    if (closest.iValues?.length) {
      setIData(closest.iValues);
    }

    if (closest.qValues?.length) {
      setQData(closest.qValues);
    }
  };

  /*
   * START CAMERA.
   */
  const startCamera =
    async (): Promise<boolean> => {
      try {
        setCameraStatus(
          "STARTING"
        );

        const stream =
          await navigator.mediaDevices.getUserMedia(
            {
              video: true,
              audio: false,
            }
          );

        cameraStreamRef.current =
          stream;

        const track =
          stream.getVideoTracks()[0];

        if (track) {
          setCameraName(
            track.label ||
              "Camera"
          );
        }

        setCameraStatus(
          "LIVE"
        );

        return true;
      } catch (error) {
        console.error(
          "[Wi-Track] Camera error:",
          error
        );

        setCameraStatus(
          "ERROR"
        );

        return false;
      }
    };

  /*
   * START VIDEO RECORDING.
   */
  const startVideoRecording =
    (): boolean => {
      const stream =
        cameraStreamRef.current;

      if (!stream) {
        return false;
      }

      try {
        let mimeType =
          "video/webm;codecs=vp9";

        if (
          !MediaRecorder.isTypeSupported(
            mimeType
          )
        ) {
          mimeType =
            "video/webm;codecs=vp8";
        }

        if (
          !MediaRecorder.isTypeSupported(
            mimeType
          )
        ) {
          mimeType =
            "video/webm";
        }

        const recorder =
          new MediaRecorder(
            stream,
            {
              mimeType,
            }
          );

        videoChunksRef.current =
          [];

        recorder.ondataavailable =
          (event) => {
            if (
              event.data.size >
              0
            ) {
              videoChunksRef.current.push(
                event.data
              );
            }
          };

        recorder.onstop = () => {
          const blob =
            new Blob(
              videoChunksRef.current,
              {
                type:
                  recorder.mimeType ||
                  "video/webm",
              }
            );

          recordedVideoRef.current =
            blob;

          setVideoStatus(
            "READY"
          );

          if (
            videoPlaybackUrl.current
          ) {
            URL.revokeObjectURL(
              videoPlaybackUrl.current
            );
          }

          videoPlaybackUrl.current =
            URL.createObjectURL(
              blob
            );
        };

        recorder.start(1000);

        mediaRecorderRef.current =
          recorder;

        setVideoStatus(
          "RECORDING"
        );

        return true;
      } catch (error) {
        console.error(
          "[Wi-Track] Video recording error:",
          error
        );

        return false;
      }
    };

  /*
   * STOP VIDEO RECORDING.
   */
  const stopVideoRecording =
    (): Promise<void> => {
      return new Promise(
        (resolve) => {
          const recorder =
            mediaRecorderRef.current;

          if (
            !recorder ||
            recorder.state ===
              "inactive"
          ) {
            resolve();
            return;
          }

          recorder.addEventListener(
            "stop",
            () => {
              resolve();
            },
            {
              once: true,
            }
          );

          setVideoStatus(
            "PROCESSING"
          );

          recorder.stop();
        }
      );
    };

  /*
   * STOP CAMERA.
   */
  const stopCamera = () => {
    cameraStreamRef.current
      ?.getTracks()
      .forEach((track) => {
        track.stop();
      });

    cameraStreamRef.current =
      null;

    setCameraStatus("OFF");

    setCameraName(
      "No camera selected"
    );
  };

  /*
   * WebSocket CSI handler.
   *
   * Later this will receive the
   * REAL backend CSI data.
   */
  const handleCSIData = (
    data: CSIData[]
  ) => {
    recordedSamples.current.push(
      ...data
    );

    setSampleCount(
      recordedSamples.current.length
    );

    if (!isRecording) {
      return;
    }

    const latest =
      data[data.length - 1];

    if (!latest) {
      return;
    }

    if (latest.iValues?.length) {
      setIData(latest.iValues);
    }

    if (latest.qValues?.length) {
      setQData(latest.qValues);
    }
  };

  /*
   * WebSocket YOLO handler.
   */
  const handleYOLOData = (
    data: YOLOData
  ) => {
    setYoloHistory((previous) => [
      ...previous,
      data,
    ]);

    setCurrentYOLO(data);

    console.log(
      "[Wi-Track] YOLO data received:",
      data
    );
  };

  /*
   * Find the YOLO result closest to the
   * selected timeline position.
   */
  const showYOLOAtTime = (
    time: number
  ) => {
    if (
      yoloHistory.length === 0 ||
      !startedAt
    ) {
      setCurrentYOLO(null);
      return;
    }

    const targetTimestamp =
      new Date(startedAt).getTime() +
      time * 1000;

    let closest =
      yoloHistory[0];

    let closestDifference =
      Math.abs(
        closest.timestamp -
          targetTimestamp
      );

    for (
      const result of yoloHistory
    ) {
      const difference =
        Math.abs(
          result.timestamp -
            targetTimestamp
        );

      if (
        difference <
        closestDifference
      ) {
        closest = result;
        closestDifference =
          difference;
      }
    }

    setCurrentYOLO(closest);
  };

  /*
   * WebSocket MODEL handler.
   */
  const handleModelData = (
    data: ModelData
  ) => {
    setModelHistory((previous) => [
      ...previous,
      data,
    ]);

    setCurrentModel(data);

    console.log(
      "[Wi-Track] MODEL data received:",
      data
    );
  };

  /*
   * Find the MODEL result nearest to the
   * selected inspection timestamp.
   */
  const showModelAtTime = (time: number) => {
    if (
      modelHistory.length === 0 ||
      !startedAt
    ) {
      setCurrentModel(null);
      return;
    }

    const targetTimestamp =
      new Date(startedAt).getTime() +
      time * 1000;

    let closest = modelHistory[0];
    let closestDifference = Math.abs(
      closest.timestamp - targetTimestamp
    );

    for (const result of modelHistory) {
      const difference = Math.abs(
        result.timestamp - targetTimestamp
      );

      if (difference < closestDifference) {
        closest = result;
        closestDifference = difference;
      }
    }

    setCurrentModel(closest);
  };

  /*
   * START EVERYTHING.
   */
  const handleStart = async () => {
    const cameraStarted =
      await startCamera();

    /*
     * IMPORTANT:
     * Timer does not start if camera
     * permission is denied.
     */
    if (!cameraStarted) {
      return;
    }

    const videoStarted =
      startVideoRecording();

    if (!videoStarted) {
      stopCamera();
      return;
    }

    const newSessionId =
      createSessionId();

    const startTime =
      new Date().toISOString();

    recordedSamples.current =
      [];

    setSessionId(
      newSessionId
    );

    setStartedAt(
      startTime
    );

    setElapsedTime(0);

    setSelectedTime(0);

    setLastSession(null);

    setIsRecording(true);

    /*
     * WebSocket connection.
     */
    setConnectionStatus(
      "CONNECTING"
    );
    /*
     * Clear results from the previous session.
     */
    setYoloHistory([]);
    setCurrentYOLO(null);
    setModelHistory([]);
    setCurrentModel(null);
    setSampleCount(0);

    const websocket =
      new CSIWebSocket(
        WS_URL,
        handleCSIData,
        handleYOLOData,
        handleModelData,
        () => {
          setConnectionStatus(
            "CONNECTED"
          );
        },
        () => {
          setConnectionStatus(
            "DISCONNECTED"
          );
        },
        () => {
          setConnectionStatus(
            "ERROR"
          );
        }
      );
    websocketRef.current =
      websocket;

    websocket.connect();
  };

  /*
   * STOP EVERYTHING.
   */
  const handleStop = async () => {
    if (
      !sessionId ||
      !startedAt
    ) {
      return;
    }

    const finalDuration =
      elapsedTime;

    await stopVideoRecording();

    const stoppedAt =
      new Date().toISOString();

    const session: RecordingSession =
      {
        sessionId,
        startedAt,
        stoppedAt,
        duration:
          finalDuration,
        samples: [
          ...recordedSamples.current,
        ],
      };

    setLastSession(
      session
    );

    /*
     * Start inspection at the END
     * of the recording.
     */
    setSelectedTime(
      finalDuration
    );

    setIsRecording(false);

    websocketRef.current?.disconnect();

    websocketRef.current =
      null;

    setConnectionStatus(
      "DISCONNECTED"
    );

    stopCamera();

    /*
     * Show CSI at final timestamp.
     */
    showCSIAtTime(
      finalDuration
    );

    showYOLOAtTime(
      finalDuration
    );

    showModelAtTime(
      finalDuration
    );
  };

  /*
   * REAL INSPECTION SLIDER.
   */
  const handleSliderChange = (
    value: number
  ) => {
    const safeValue =
      Math.max(
        0,
        Math.min(
          value,
          elapsedTime
        )
      );

    setSelectedTime(
      safeValue
    );

    /*
     * Move recorded video.
     */
    if (
      !isRecording &&
      videoRef.current &&
      videoPlaybackUrl.current
    ) {
      videoRef.current.currentTime =
        safeValue;
    }

    /*
     * Move CSI graph.
     */
    if (!isRecording) {
      showCSIAtTime(
        safeValue
      );

      showYOLOAtTime(
        safeValue
      );

      showModelAtTime(
        safeValue
      );
    }
  };

  /*
   * Download CSI CSV generated
   * from the actual recorded session.
   */
  const handleDownload =
    async () => {
      if (!lastSession) {
        return;
      }

      const sessionStart =
        new Date(
          lastSession.startedAt
        ).getTime();

      /*
       * Build a complete CSI CSV.
       *
       * If the backend sends full I/Q arrays,
       * we preserve every subcarrier.
       */
      const maxIValues =
        lastSession.samples.reduce(
          (
            maximum,
            sample
          ) =>
            Math.max(
              maximum,
              sample.iValues
                ?.length ?? 0
            ),
          0
        );

      const maxQValues =
        lastSession.samples.reduce(
          (
            maximum,
            sample
          ) =>
            Math.max(
              maximum,
              sample.qValues
                ?.length ?? 0
            ),
          0
        );

      const iCount =
        Math.max(
          maxIValues,
          1
        );

      const qCount =
        Math.max(
          maxQValues,
          1
        );

      const csiHeader = [
        "session_id",
        "timestamp",
        "elapsed_ms",
        ...Array.from(
          {
            length: iCount,
          },
          (_, index) =>
            `I${index}`
        ),
        ...Array.from(
          {
            length: qCount,
          },
          (_, index) =>
            `Q${index}`
        ),
      ];

      const csiRows =
        lastSession.samples.map(
          (sample) => {
            const elapsedMs =
              sample.timestamp -
              sessionStart;

            const iValues =
              sample.iValues ??
              [sample.i];

            const qValues =
              sample.qValues ??
              [sample.q];

            return [
              lastSession.sessionId,
              new Date(
                sample.timestamp
              ).toISOString(),
              elapsedMs,
              ...Array.from(
                {
                  length: iCount,
                },
                (_, index) =>
                  iValues[index] ??
                  ""
              ),
              ...Array.from(
                {
                  length: qCount,
                },
                (_, index) =>
                  qValues[index] ??
                  ""
              ),
            ].join(",");
          }
        );

      const csiContent = [
        csiHeader.join(","),
        ...csiRows,
      ].join("\n");

      /*
       * Metadata about the complete session.
       */
      const metadata = {
        project: "Wi-Track",
        sessionId:
          lastSession.sessionId,
        startedAt:
          lastSession.startedAt,
        stoppedAt:
          lastSession.stoppedAt,
        durationSeconds:
          lastSession.duration,
        csiSamples:
          lastSession.samples.length,
        yoloResults:
          yoloHistory.length,
        modelResults:
          modelHistory.length,
        files: [
          "csi_data.csv",
          "video.webm",
          "yolo_results.json",
          "model_results.json",
        ],
      };

      const entries: DownloadEntry[] =
        [
          {
            name: "session_metadata.json",
            data: textEncoder.encode(
              JSON.stringify(
                metadata,
                null,
                2
              )
            ),
          },
          {
            name: "csi_data.csv",
            data: textEncoder.encode(
              csiContent
            ),
          },
          {
            name: "yolo_results.json",
            data: textEncoder.encode(
              JSON.stringify(
                yoloHistory,
                null,
                2
              )
            ),
          },
          {
            name: "model_results.json",
            data: textEncoder.encode(
              JSON.stringify(
                modelHistory,
                null,
                2
              )
            ),
          },
        ];

      /*
       * Add the recorded camera video when
       * it exists.
       */
      if (
        recordedVideoRef.current
      ) {
        entries.push({
          name: "video.webm",
          data:
            await blobToUint8Array(
              recordedVideoRef.current
            ),
        });
      }

      const zipBlob =
        createZip(entries);

      const url =
        URL.createObjectURL(
          zipBlob
        );

      const link =
        document.createElement(
          "a"
        );

      link.href = url;

      link.download =
        `${lastSession.sessionId}.zip`;

      document.body.appendChild(
        link
      );

      link.click();

      link.remove();

      window.setTimeout(() => {
        URL.revokeObjectURL(
          url
        );
      }, 1000);
    };


  /*
   * Draw I/Q values.
   */
  const createPolyline = (
    values: number[],
    width: number,
    height: number
  ) => {
    if (
      values.length < 2
    ) {
      return "";
    }

    const min =
      Math.min(...values);

    const max =
      Math.max(...values);

    const range =
      max - min === 0
        ? 1
        : max - min;

    return values
      .map(
        (value, index) => {
          const x =
            (index /
              (values.length -
                1)) *
            width;

          const y =
            height -
            ((value - min) /
              range) *
              height;

          return `${x},${y}`;
        }
      )
      .join(" ");
  };

  const iPoints = useMemo(
    () =>
      createPolyline(
        iData,
        100,
        100
      ),
    [iData]
  );

  const qPoints = useMemo(
    () =>
      createPolyline(
        qData,
        100,
        100
      ),
    [qData]
  );

  /*
   * Timeline.
   */
  const timelineRange =
    getTimelineRange(
      elapsedTime
    );

  const timelineStep =
    getTimelineStep(
      timelineRange
    );

  const sliderProgress =
    timelineRange > 0
      ? Math.min(
          (selectedTime /
            timelineRange) *
            100,
          100
        )
      : 0;

  const timelineLabels: number[] =
    [];

  for (
    let time = 0;
    time <= timelineRange;
    time += timelineStep
  ) {
    timelineLabels.push(
      time
    );
  }

  /*
   * Camera should show:
   *
   * LIVE during recording
   * PLAYBACK after recording
   */
  const showCameraVideo =
    cameraStatus === "LIVE" ||
    Boolean(
      videoPlaybackUrl.current
    );

  /*
   * Current YOLO detection for the camera
   * overlay. This remains inactive until
   * the backend sends bounding-box coordinates.
   */
  const currentDetection =
    currentYOLO?.detections?.length
      ? (currentYOLO.detections[0] as unknown as YOLODetectionView)
      : null;

  const videoElement =
    videoRef.current;

  const boxLeft = toPercent(
    currentDetection?.x,
    videoElement?.videoWidth
  );

  const boxTop = toPercent(
    currentDetection?.y,
    videoElement?.videoHeight
  );

  const boxWidth = toPercent(
    currentDetection?.width,
    videoElement?.videoWidth
  );

  const boxHeight = toPercent(
    currentDetection?.height,
    videoElement?.videoHeight
  );

  const hasYOLOBox =
    boxLeft !== undefined &&
    boxTop !== undefined &&
    boxWidth !== undefined &&
    boxHeight !== undefined;

  return (
    <div className="witrack-dashboard">
      {/* HEADER */}
      <header className="witrack-header">
        <div className="witrack-logo">
          WI-TRACK
        </div>

        <div className="witrack-status">
          <span
            className={`status-dot ${
              isRecording
                ? "status-recording"
                : ""
            }`}
          ></span>

          {isRecording
            ? "RECORDING"
            : "SYSTEM READY"}
        </div>
      </header>

      <main className="witrack-main">
        <section className="top-section">
          <div className="visualization-area">
            {/* CAMERA */}
            <div className="panel camera-panel">
              <div className="panel-header">
                <span>
                  CAMERA
                </span>

                <span className="panel-status">
                  {isRecording
                    ? "LIVE"
                    : videoPlaybackUrl.current
                      ? "PLAYBACK"
                      : "READY"}
                </span>
              </div>

              <div className="camera-content">
                {showCameraVideo ? (
                  <>
                    <div
                      style={{
                        position:
                          "relative",
                        width: "100%",
                        height: "100%",
                      }}
                    >
                      <video
                        ref={videoRef}
                        className="camera-video"
                        autoPlay={
                          isRecording
                        }
                        muted
                        playsInline
                        controls={
                          !isRecording
                        }
                      />

                      {hasYOLOBox && (
                        <div
                          style={{
                            position:
                              "absolute",
                            left: `${boxLeft}%`,
                            top: `${boxTop}%`,
                            width: `${boxWidth}%`,
                            height: `${boxHeight}%`,
                            border:
                              "2px solid #39ff88",
                            pointerEvents:
                              "none",
                            boxSizing:
                              "border-box",
                          }}
                        >
                          <span
                            style={{
                              position:
                                "absolute",
                              left: 0,
                              top: "-24px",
                              padding:
                                "3px 7px",
                              fontSize:
                                "11px",
                              lineHeight:
                                "16px",
                              background:
                                "#39ff88",
                              color:
                                "#07110b",
                              fontWeight:
                                700,
                              whiteSpace:
                                "nowrap",
                            }}
                          >
                            {currentDetection?.label ??
                              "object"}{" "}
                            {currentDetection?.confidence !==
                            undefined
                              ? `${(
                                  currentDetection.confidence *
                                  100
                                ).toFixed(1)}%`
                              : ""}
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="camera-overlay">
                      <span className="camera-live-indicator">
                        <i></i>
                        {isRecording
                          ? "LIVE"
                          : "PLAYBACK"}
                      </span>

                      <span className="camera-name">
                        {cameraName}
                      </span>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="camera-icon">
                      ◉
                    </div>

                    <span>
                      VIDEO FEED
                    </span>

                    <small>
                      {cameraStatus ===
                      "STARTING"
                        ? "Starting camera..."
                        : cameraStatus ===
                            "ERROR"
                          ? "Camera access unavailable"
                          : "Camera stream will appear here"}
                    </small>
                  </>
                )}
              </div>
            </div>

            {/* CSI GRAPH */}
            <div className="panel csi-panel">
              <div className="panel-header">
                <span>
                  CSI GRAPH
                </span>

                <div className="graph-legend">
                  <span>
                    <i className="legend-i"></i>
                    I
                  </span>

                  <span>
                    <i className="legend-q"></i>
                    Q
                  </span>
                </div>
              </div>

              <div className="csi-graph">
                <div className="graph-grid"></div>

                <div className="graph-label label-i">
                  I
                </div>

                <div className="graph-label label-q">
                  Q
                </div>

                <svg
                  className="csi-wave-svg"
                  viewBox="0 0 100 100"
                  preserveAspectRatio="none"
                >
                  {iPoints && (
                    <polyline
                      points={
                        iPoints
                      }
                      className="i-wave-line"
                    />
                  )}

                  {qPoints && (
                    <polyline
                      points={
                        qPoints
                      }
                      className="q-wave-line"
                    />
                  )}
                </svg>

                {iData.length ===
                  0 &&
                  qData.length ===
                    0 && (
                    <div className="graph-empty">
                      {connectionStatus === "CONNECTED"
                        ? "WAITING FOR CSI DATA"
                        : "NO CSI DATA"}
                    </div>
                  )}
              </div>

              <div
                style={{
                  padding:
                    "6px 12px",
                  fontSize:
                    "10px",
                  opacity: 0.6,
                }}
              >
                {sampleCount > 0
                  ? `BACKEND CSI • ${
                      sampleCount
                    } frames • ${
                      iData.length
                    } I + ${
                      qData.length
                    } Q values`
                  : "WAITING FOR BACKEND CSI"}
              </div>
            </div>

            {/* TIMELINE */}
            <div className="panel timeline-panel">
              <div className="panel-header">
                <span>
                  TIMELINE
                </span>

                <span className="timestamp">
                  {formatTime(
                    selectedTime
                  )}
                </span>
              </div>

              <div className="timeline-container">
                <div className="timeline-line"></div>

                <div
                  className="timeline-progress"
                  style={{
                    width: `${sliderProgress}%`,
                  }}
                ></div>

                <input
                  className="timeline-slider"
                  type="range"
                  min="0"
                  max={Math.max(
                    timelineRange,
                    1
                  )}
                  step="1"
                  value={Math.min(
                    selectedTime,
                    timelineRange
                  )}
                  onChange={(event) =>
                    handleSliderChange(
                      Number(
                        event.target
                          .value
                      )
                    )
                  }
                />

                <div className="timeline-labels">
                  {timelineLabels.map(
                    (time) => (
                      <span
                        key={time}
                      >
                        {time}s
                      </span>
                    )
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* CONTROLS */}
          <aside className="control-panel">
            <button
              className="control-button start-button"
              onClick={
                handleStart
              }
              disabled={
                isRecording
              }
            >
              <span>▶</span>
              START
            </button>

            <button
              className="control-button stop-button"
              onClick={
                handleStop
              }
              disabled={
                !isRecording
              }
            >
              <span>■</span>
              STOP
            </button>

            <button
              className="control-button"
              type="button"
            >
              <span>◆</span>
              MODEL
            </button>

            <button
              className="control-button"
              type="button"
            >
              <span>◇</span>
              YOLO
            </button>

            <button
              className="control-button download-button"
              type="button"
              onClick={
                handleDownload
              }
              disabled={
                !lastSession ||
                isRecording
              }
            >
              <span>↓</span>
              DOWNLOAD
            </button>
          </aside>
        </section>

        {/* OUTPUTS */}
        <section className="output-section">
          <div className="output-panel">
            <div className="output-header">
              MODEL OUTPUT
            </div>

            <div className="output-content">
              <div className="output-row">
                <span>
                  Prediction
                </span>

                <strong>
                  {currentModel?.prediction ??
                    "---"}
                </strong>
              </div>

              <div className="output-row">
                <span>
                  Confidence
                </span>

                <strong>
                  {currentModel?.confidence !==
                  undefined
                    ? `${(
                        currentModel.confidence *
                        100
                      ).toFixed(1)}%`
                    : "---"}
                </strong>
              </div>

              <div className="output-row">
                <span>
                  Status
                </span>

                <strong
                  className={
                    currentModel?.status ===
                    "DETECTED"
                      ? "active-status"
                      : "waiting"
                  }
                >
                  {currentModel?.status ??
                    "WAITING"}
                </strong>
              </div>
            </div>
          </div>

          <div className="output-panel">
            <div className="output-header">
              YOLO OUTPUT
            </div>

            <div className="output-content">
              <div className="output-row">
                <span>
                  Detection
                </span>

                <strong>
                  {currentYOLO?.detections?.length
                    ? currentYOLO.detections[0]
                        .label
                    : "---"}
                </strong>
              </div>

              <div className="output-row">
                <span>
                  Confidence
                </span>

                <strong>
                  {currentYOLO?.detections?.length
                    ? `${(
                        currentYOLO.detections[0]
                          .confidence * 100
                      ).toFixed(1)}%`
                    : "---"}
                </strong>
              </div>

              <div className="output-row">
                <span>
                  Status
                </span>

                <strong
                  className={
                    currentYOLO?.detections?.length
                      ? "active-status"
                      : "waiting"
                  }
                >
                  {currentYOLO?.detections?.length
                    ? "DETECTED"
                    : "WAITING"}
                </strong>
              </div>
            </div>
          </div>
        </section>

        {/* SESSION */}
        {sessionId && (
          <section className="session-info">
            <span>
              SESSION
            </span>

            <strong>
              {sessionId}
            </strong>

            <span>
              WS:{" "}
              {connectionStatus}
            </span>

            <span>
              CSI FRAMES:{" "}
              {sampleCount}
            </span>

            <span>
              CAMERA:{" "}
              {cameraStatus}
            </span>

            <span>
              VIDEO:{" "}
              {videoStatus}
            </span>

            <span>
              SELECTED:{" "}
              {formatTime(
                selectedTime
              )}
            </span>
          </section>
        )}
      </main>
    </div>
  );
}

export default Dashboard;