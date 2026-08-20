import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import "./piyush.css";
import csiCsvUrl from "./csi_data.csv?url";
import type {
  CSIData,
  YOLOData,
  ModelData,
} from "./websocket";

const RECORDING_FPS = 30;

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

interface SampleCSIFrame {
  iValues: number[];
  qValues: number[];
  label?: string;
}

let sampleCSIFramesLoader:
  | Promise<SampleCSIFrame[]>
  | null = null;

function parseSampleCSI(
  csvText: string
) {
  const lines =
    csvText
      .trim()
      .split(/\r?\n/)
      .filter(Boolean);

  const headers =
    lines[0]?.split(",") ?? [];

  const labelIndex = headers.indexOf("label");

  const iColumns =
    headers.flatMap(
      (header, index) =>
        /^I\d+$/.test(header)
          ? [index]
          : []
    );

  const qColumns =
    headers.flatMap(
      (header, index) =>
        /^Q\d+$/.test(header)
          ? [index]
          : []
    );

  return lines
    .slice(1)
    .flatMap((line) => {
      const cells =
        line.split(",");

      const iValues =
        iColumns.map(
          (index) =>
            Number(cells[index]) || 0
        );

      const qValues =
        qColumns.map(
          (index) =>
            Number(cells[index]) || 0
        );

      if (
        iValues.length === 0 ||
        qValues.length === 0
      ) {
        return [];
      }

      const label = labelIndex !== -1 ? cells[labelIndex] : undefined;

      return [
        {
          iValues,
          qValues,
          label,
        },
      ];
    });
}

async function loadSampleCSIFrames() {
  if (!sampleCSIFramesLoader) {
    sampleCSIFramesLoader =
      fetch(csiCsvUrl)
        .then((response) => {
          if (!response.ok) {
            throw new Error(
              "Failed to load sample CSI CSV"
            );
          }

          return response.text();
        })
        .then(parseSampleCSI);
  }

  return sampleCSIFramesLoader;
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

type OrtTensorData = Float32Array;

interface OrtTensor {
  data: OrtTensorData;
  dims: number[];
}

interface OrtSession {
  inputNames: string[];
  outputNames: string[];
  run(
    feeds: Record<string, OrtTensor>
  ): Promise<Record<string, OrtTensor>>;
}

interface OrtRuntime {
  env: {
    wasm: {
      wasmPaths?: string;
    };
  };
  Tensor: new (
    type: "float32",
    data: Float32Array,
    dims: number[]
  ) => OrtTensor;
  InferenceSession: {
    create(
      modelPath: string,
      options?: {
        executionProviders?: string[];
      }
    ): Promise<OrtSession>;
  };
}

interface FFmpegInstance {
  loaded?: boolean;
  load(options: {
    coreURL: string;
    wasmURL: string;
    classWorkerURL?: string;
  }): Promise<void>;
  writeFile(
    path: string,
    data: Uint8Array
  ): Promise<void>;
  readFile(path: string): Promise<
    Uint8Array | string
  >;
  deleteFile(path: string): Promise<void>;
  exec(args: string[]): Promise<number>;
  on(
    event: "log",
    handler: (data: {
      message: string;
    }) => void
  ): void;
}

interface FFmpegModule {
  FFmpeg: new () => FFmpegInstance;
}

declare global {
  interface Window {
    ort?: OrtRuntime;
    FFmpegWASM?: FFmpegModule;
    FFmpeg?: FFmpegModule;
  }
}

const YOLO_MODEL_PATH =
  "/yolov8n.onnx";

const YOLO_SIZE = 640;

const YOLO_CONFIDENCE_THRESHOLD =
  0.35;

const YOLO_IOU_THRESHOLD =
  0.45;

const COCO_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

let ortLoader: Promise<OrtRuntime> | null =
  null;

let ffmpegLoader:
  | Promise<FFmpegInstance>
  | null = null;

function loadScript(
  src: string
): Promise<void> {
  return new Promise(
    (resolve, reject) => {
      const existing =
        document.querySelector(
          `script[src="${src}"]`
        );

      if (existing) {
        resolve();
        return;
      }

      const script =
        document.createElement(
          "script"
        );

      script.src = src;
      script.async = true;

      script.onload = () => {
        resolve();
      };

      script.onerror = () => {
        reject(
          new Error(
            `Failed to load ${src}`
          )
        );
      };

      document.head.appendChild(
        script
      );
    }
  );
}

async function toBlobURL(
  url: string,
  mimeType: string
) {
  const response =
    await fetch(url);

  if (!response.ok) {
    throw new Error(
      `Failed to load ${url}`
    );
  }

  const blob =
    await response.blob();

  return URL.createObjectURL(
    new Blob([blob], {
      type: mimeType,
    })
  );
}

async function loadOrtRuntime() {
  if (!ortLoader) {
    ortLoader = (async () => {
      await loadScript(
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js"
      );

      const ort = window.ort;

      if (!ort) {
        throw new Error(
          "ONNX Runtime failed to initialize"
        );
      }

      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

      return ort;
    })();
  }

  return ortLoader;
}

async function loadFFmpeg() {
  if (!ffmpegLoader) {
    ffmpegLoader = (async () => {
      await loadScript(
        "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/umd/ffmpeg.js"
      );

      const ffmpegModule =
        window.FFmpegWASM ??
        window.FFmpeg;

      if (!ffmpegModule) {
        throw new Error(
          "ffmpeg.wasm failed to initialize"
        );
      }

      const ffmpeg =
        new ffmpegModule.FFmpeg();

      ffmpeg.on(
        "log",
        ({ message }) => {
          console.log(
            "[Wi-Track] ffmpeg:",
            message
          );
        }
      );

      const ffmpegBaseURL =
        "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/umd";

      const coreBaseURL =
        "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/umd";

      await ffmpeg.load({
        classWorkerURL:
          await toBlobURL(
            `${ffmpegBaseURL}/814.ffmpeg.js`,
            "text/javascript"
          ),
        coreURL: `${coreBaseURL}/ffmpeg-core.js`,
        wasmURL: `${coreBaseURL}/ffmpeg-core.wasm`,
      });

      return ffmpeg;
    })().catch((error) => {
      /*
       * Reset so the next attempt
       * retries the load instead of
       * returning the same rejection.
       */
      ffmpegLoader = null;
      throw error;
    });
  }

  return ffmpegLoader;
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

function createYOLOInput(
  video: HTMLVideoElement
) {
  const canvas =
    document.createElement(
      "canvas"
    );

  canvas.width = YOLO_SIZE;
  canvas.height = YOLO_SIZE;

  const context =
    canvas.getContext("2d");

  if (!context) {
    throw new Error(
      "Could not prepare YOLO canvas"
    );
  }

  const sourceWidth =
    video.videoWidth || 1;

  const sourceHeight =
    video.videoHeight || 1;

  const scale =
    Math.min(
      YOLO_SIZE / sourceWidth,
      YOLO_SIZE / sourceHeight
    );

  const drawWidth =
    sourceWidth * scale;

  const drawHeight =
    sourceHeight * scale;

  const padX =
    (YOLO_SIZE - drawWidth) / 2;

  const padY =
    (YOLO_SIZE - drawHeight) / 2;

  context.fillStyle = "#727272";
  context.fillRect(
    0,
    0,
    YOLO_SIZE,
    YOLO_SIZE
  );

  context.drawImage(
    video,
    padX,
    padY,
    drawWidth,
    drawHeight
  );

  const pixels =
    context.getImageData(
      0,
      0,
      YOLO_SIZE,
      YOLO_SIZE
    ).data;

  const input =
    new Float32Array(
      3 * YOLO_SIZE * YOLO_SIZE
    );

  const planeSize =
    YOLO_SIZE * YOLO_SIZE;

  for (
    let pixelIndex = 0;
    pixelIndex < planeSize;
    pixelIndex += 1
  ) {
    const sourceIndex =
      pixelIndex * 4;

    input[pixelIndex] =
      pixels[sourceIndex] / 255;

    input[
      planeSize + pixelIndex
    ] =
      pixels[
        sourceIndex + 1
      ] / 255;

    input[
      planeSize * 2 +
        pixelIndex
    ] =
      pixels[
        sourceIndex + 2
      ] / 255;
  }

  return {
    input,
    sourceWidth,
    sourceHeight,
    scale,
    padX,
    padY,
  };
}

function boxIou(
  a: YOLODetectionView,
  b: YOLODetectionView
) {
  const ax1 = a.x ?? 0;
  const ay1 = a.y ?? 0;
  const ax2 =
    ax1 + (a.width ?? 0);
  const ay2 =
    ay1 + (a.height ?? 0);

  const bx1 = b.x ?? 0;
  const by1 = b.y ?? 0;
  const bx2 =
    bx1 + (b.width ?? 0);
  const by2 =
    by1 + (b.height ?? 0);

  const intersectionWidth =
    Math.max(
      0,
      Math.min(ax2, bx2) -
        Math.max(ax1, bx1)
    );

  const intersectionHeight =
    Math.max(
      0,
      Math.min(ay2, by2) -
        Math.max(ay1, by1)
    );

  const intersection =
    intersectionWidth *
    intersectionHeight;

  const union =
    (a.width ?? 0) *
      (a.height ?? 0) +
    (b.width ?? 0) *
      (b.height ?? 0) -
    intersection;

  return union <= 0
    ? 0
    : intersection / union;
}

function nonMaxSuppression(
  detections: YOLODetectionView[]
) {
  const sorted = [
    ...detections,
  ].sort(
    (a, b) =>
      b.confidence -
      a.confidence
  );

  const selected: YOLODetectionView[] =
    [];

  for (const detection of sorted) {
    const overlaps =
      selected.some(
        (picked) =>
          detection.label ===
            picked.label &&
          boxIou(
            detection,
            picked
          ) > YOLO_IOU_THRESHOLD
      );

    if (!overlaps) {
      selected.push(
        detection
      );
    }

    if (selected.length >= 8) {
      break;
    }
  }

  return selected;
}

function parseYOLOOutput(
  tensor: OrtTensor,
  meta: ReturnType<
    typeof createYOLOInput
  >
) {
  const data = tensor.data;
  const dims = tensor.dims;

  if (dims.length < 3) {
    return [];
  }

  const first = dims[1];
  const second = dims[2];
  const transposed =
    first < second;

  const boxCount =
    transposed ? second : first;

  const attributes =
    transposed ? first : second;

  const classOffset =
    attributes > 84 ? 5 : 4;

  const classCount =
    attributes - classOffset;

  const detections: YOLODetectionView[] =
    [];

  for (
    let boxIndex = 0;
    boxIndex < boxCount;
    boxIndex += 1
  ) {
    const valueAt = (
      attribute: number
    ) =>
      transposed
        ? data[
            attribute *
              boxCount +
              boxIndex
          ]
        : data[
            boxIndex *
              attributes +
              attribute
          ];

    const objectness =
      classOffset === 5
        ? valueAt(4)
        : 1;

    let bestClass = 0;
    let bestScore = 0;

    for (
      let classIndex = 0;
      classIndex < classCount;
      classIndex += 1
    ) {
      const score =
        valueAt(
          classOffset +
            classIndex
        ) * objectness;

      if (score > bestScore) {
        bestScore = score;
        bestClass = classIndex;
      }
    }

    if (
      bestScore <
      YOLO_CONFIDENCE_THRESHOLD
    ) {
      continue;
    }

    const centerX =
      valueAt(0);
    const centerY =
      valueAt(1);
    const width = valueAt(2);
    const height = valueAt(3);

    const x =
      (centerX -
        width / 2 -
        meta.padX) /
      meta.scale;

    const y =
      (centerY -
        height / 2 -
        meta.padY) /
      meta.scale;

    detections.push({
      label:
        COCO_CLASSES[
          bestClass
        ] ?? `class ${bestClass}`,
      confidence:
        bestScore,
      x: Math.max(
        0,
        Math.min(
          x,
          meta.sourceWidth
        )
      ),
      y: Math.max(
        0,
        Math.min(
          y,
          meta.sourceHeight
        )
      ),
      width: Math.max(
        0,
        Math.min(
          width / meta.scale,
          meta.sourceWidth
        )
      ),
      height: Math.max(
        0,
        Math.min(
          height / meta.scale,
          meta.sourceHeight
        )
      ),
    });
  }

  return nonMaxSuppression(
    detections
  );
}

function drawGraphCanvas(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  iValues: number[],
  qValues: number[]
) {
  context.fillStyle = "#0d1219";
  context.fillRect(
    0,
    0,
    width,
    height
  );

  context.strokeStyle = "#1b222c";
  context.lineWidth = 1;

  for (
    let x = 40;
    x < width;
    x += 40
  ) {
    context.beginPath();
    context.moveTo(x, 48);
    context.lineTo(
      x,
      height - 42
    );
    context.stroke();
  }

  for (
    let y = 48;
    y < height - 42;
    y += 40
  ) {
    context.beginPath();
    context.moveTo(28, y);
    context.lineTo(
      width - 24,
      y
    );
    context.stroke();
  }

  context.fillStyle = "#aeb8c7";
  context.font =
    "700 22px system-ui";
  context.fillText(
    "CSI GRAPH",
    28,
    34
  );

  context.font =
    "700 16px system-ui";
  context.fillStyle = "#60a5fa";
  context.fillText("I", 28, 90);

  context.fillStyle = "#c084fc";
  context.fillText(
    "Q",
    28,
    134
  );

  const drawWave = (
    values: number[],
    color: string
  ) => {
    if (values.length < 2) {
      return;
    }

    const min =
      Math.min(...values);

    const max =
      Math.max(...values);

    const range =
      max - min || 1;

    const plotLeft = 48;
    const plotTop = 58;
    const plotWidth =
      width - 78;
    const plotHeight =
      height - 112;

    context.beginPath();
    context.strokeStyle = color;
    context.lineWidth = 2;

    values.forEach(
      (value, index) => {
        const x =
          plotLeft +
          (index /
            (values.length -
              1)) *
            plotWidth;

        const y =
          plotTop +
          plotHeight -
          ((value - min) /
            range) *
            plotHeight;

        if (index === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      }
    );

    context.stroke();
  };

  drawWave(iValues, "#60a5fa");
  drawWave(qValues, "#c084fc");
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

  const [yoloEnabled, setYoloEnabled] =
    useState(true);

  const [modelEnabled, setModelEnabled] =
    useState(true);

  const [videoStatus, setVideoStatus] =
    useState<
      | "IDLE"
      | "RECORDING"
      | "PROCESSING"
      | "READY"
    >("IDLE");

  /*
   * Current I/Q values from the
   * sample CSI stream.
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

  const sampleIndexRef =
    useRef(0);

  const sampleFramesRef =
    useRef<SampleCSIFrame[]>([]);

  const videoRef =
    useRef<HTMLVideoElement | null>(null);

  const sourceVideoRef =
    useRef<HTMLVideoElement | null>(
      null
    );

  const cameraStreamRef =
    useRef<MediaStream | null>(null);

  const mediaRecorderRef =
    useRef<MediaRecorder | null>(null);

  const graphRecorderRef =
    useRef<MediaRecorder | null>(null);

  const recordedVideoRef =
    useRef<Blob | null>(null);

  const recordedGraphRef =
    useRef<Blob | null>(null);

  const videoPlaybackUrl =
    useRef<string | null>(null);

  const cameraCanvasRef =
    useRef<HTMLCanvasElement | null>(
      null
    );

  const graphCanvasRef =
    useRef<HTMLCanvasElement | null>(
      null
    );

  const cameraChunksRef =
    useRef<Blob[]>([]);

  const graphChunksRef =
    useRef<Blob[]>([]);

  const renderFrameRef =
    useRef<number | null>(null);

  const yoloSessionRef =
    useRef<OrtSession | null>(
      null
    );

  const yoloLoopRef =
    useRef<number | null>(null);

  const yoloRunningRef =
    useRef(false);

  const iDataRef =
    useRef<number[]>([]);

  const qDataRef =
    useRef<number[]>([]);

  const currentYOLORef =
    useRef<YOLOData | null>(
      null
    );

  const isRecordingRef =
    useRef(false);

  const modelEnabledRef =
    useRef(true);

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
      cameraStreamRef.current
        ?.getTracks()
        .forEach((track) => {
          track.stop();
        });

      if (sourceVideoRef.current) {
        sourceVideoRef.current.pause();
        sourceVideoRef.current.srcObject =
          null;
      }

      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current
          .state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }

      if (
        graphRecorderRef.current &&
        graphRecorderRef.current
          .state !== "inactive"
      ) {
        graphRecorderRef.current.stop();
      }

      if (
        renderFrameRef.current !==
        null
      ) {
        window.clearInterval(
          renderFrameRef.current
        );
      }

      if (
        yoloLoopRef.current !==
        null
      ) {
        window.clearTimeout(
          yoloLoopRef.current
        );
      }

      yoloRunningRef.current =
        false;

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
   * Find the sample CSI frame nearest to
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
      iDataRef.current =
        closest.iValues;

      setIData(closest.iValues);
    }

    if (closest.qValues?.length) {
      qDataRef.current =
        closest.qValues;

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

        const sourceVideo =
          document.createElement(
            "video"
          );

        sourceVideo.muted = true;
        sourceVideo.playsInline = true;
        sourceVideo.srcObject = stream;

        await sourceVideo
          .play()
          .catch(() => {});

        sourceVideoRef.current =
          sourceVideo;

        if (videoRef.current) {
          videoRef.current.srcObject =
            stream;

          await videoRef.current
            .play()
            .catch(() => {});
        }

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

  const drawCameraCanvas = () => {
    const canvas =
      cameraCanvasRef.current;

    const video =
      sourceVideoRef.current ??
      videoRef.current;

    if (
      !canvas ||
      !video
    ) {
      return;
    }

    const context =
      canvas.getContext("2d");

    if (!context) {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;

    context.fillStyle = "#05070a";
    context.fillRect(
      0,
      0,
      width,
      height
    );

    const sourceWidth =
      video.videoWidth || width;

    const sourceHeight =
      video.videoHeight || height;

    const scale =
      Math.max(
        width / sourceWidth,
        height / sourceHeight
      );

    const drawWidth =
      sourceWidth * scale;

    const drawHeight =
      sourceHeight * scale;

    const offsetX =
      (width - drawWidth) / 2;

    const offsetY =
      (height - drawHeight) / 2;

    if (video.readyState >= 2) {
      context.drawImage(
        video,
        offsetX,
        offsetY,
        drawWidth,
        drawHeight
      );
    }

    const detection =
      currentYOLORef.current
        ?.detections?.[0] as
        | YOLODetectionView
        | undefined;

    if (
      detection?.x !==
        undefined &&
      detection.y !==
        undefined &&
      detection.width !==
        undefined &&
      detection.height !==
        undefined
    ) {
      const x =
        offsetX +
        detection.x * scale;

      const y =
        offsetY +
        detection.y * scale;

      const boxWidth =
        detection.width * scale;

      const boxHeight =
        detection.height * scale;

      context.strokeStyle =
        "#39ff88";
      context.lineWidth = 4;
      context.strokeRect(
        x,
        y,
        boxWidth,
        boxHeight
      );

      const label =
        `${detection.label} ${(
          detection.confidence *
          100
        ).toFixed(1)}%`;

      context.font =
        "700 18px system-ui";

      const labelWidth =
        context.measureText(
          label
        ).width + 18;

      const labelY =
        Math.max(0, y - 30);

      context.fillStyle =
        "#39ff88";
      context.fillRect(
        x,
        labelY,
        labelWidth,
        28
      );

      context.fillStyle =
        "#07110b";
      context.fillText(
        label,
        x + 9,
        labelY + 20
      );
    }

    context.fillStyle =
      "rgba(0, 0, 0, 0.55)";
    context.fillRect(
      0,
      height - 42,
      width,
      42
    );

    context.fillStyle = "#e8edf5";
    context.font =
      "700 18px system-ui";
    context.fillText(
      "CAMERA",
      22,
      height - 16
    );
  };

  const acceptNextCSIFrame =
    () => {
      const frames =
        sampleFramesRef.current;

      if (
        !isRecordingRef.current ||
        frames.length === 0
      ) {
        return;
      }

      const frame =
        frames[
          sampleIndexRef.current %
            frames.length
        ];

      sampleIndexRef.current += 1;

      const timestamp = Date.now();

      handleCSIData([
        {
          timestamp,
          i: frame.iValues[0] ?? 0,
          q: frame.qValues[0] ?? 0,
          iValues: frame.iValues,
          qValues: frame.qValues,
        },
      ]);

      if (modelEnabledRef.current && frame.label) {
        handleModelData({
          timestamp,
          prediction: frame.label,
          confidence: 0.95,
          status: frame.label !== "0" ? "DETECTED" : "WAITING",
        });
      }
    };

  const drawRecordingCanvases =
    () => {
      acceptNextCSIFrame();

      drawCameraCanvas();

      const graphCanvas =
        graphCanvasRef.current;

      const graphContext =
        graphCanvas?.getContext(
          "2d"
        );

      if (
        graphCanvas &&
        graphContext
      ) {
        drawGraphCanvas(
          graphContext,
          graphCanvas.width,
          graphCanvas.height,
          iDataRef.current,
          qDataRef.current
        );
      }

    };

  const startRenderLoop = () => {
    if (
      renderFrameRef.current !==
      null
    ) {
      window.clearInterval(
        renderFrameRef.current
      );
    }

    drawRecordingCanvases();

    renderFrameRef.current =
      window.setInterval(
        drawRecordingCanvases,
        1000 / RECORDING_FPS
      );
  };

  const stopRenderLoop = () => {
    if (
      renderFrameRef.current !==
      null
    ) {
      window.clearInterval(
        renderFrameRef.current
      );

      renderFrameRef.current =
        null;
    }
  };

  const startYOLO = () => {
    yoloRunningRef.current =
      true;

    const run = async () => {
      if (
        !yoloRunningRef.current
      ) {
        return;
      }

      try {
        const video =
          sourceVideoRef.current ??
          videoRef.current;

        if (
          video &&
          video.readyState >= 2
        ) {
          const ort =
            await loadOrtRuntime();

          if (
            !yoloSessionRef.current
          ) {
            yoloSessionRef.current =
              await ort.InferenceSession.create(
                YOLO_MODEL_PATH,
                {
                  executionProviders:
                    ["wasm"],
                }
              );
          }

          const session =
            yoloSessionRef.current;

          const meta =
            createYOLOInput(
              video
            );

          const inputName =
            session.inputNames[0];

          const outputName =
            session.outputNames[0];

          const tensor =
            new ort.Tensor(
              "float32",
              meta.input,
              [
                1,
                3,
                YOLO_SIZE,
                YOLO_SIZE,
              ]
            );

          const result =
            await session.run({
              [inputName]: tensor,
            });

          const detections =
            parseYOLOOutput(
              result[outputName],
              meta
            );

          const payload: YOLOData =
            {
              timestamp:
                Date.now(),
              detections,
            };

          handleYOLOData(payload);
        }
      } catch (error) {
        console.error(
          "[Wi-Track] YOLO inference error:",
          error
        );
      }

      if (
        yoloRunningRef.current
      ) {
        yoloLoopRef.current =
          window.setTimeout(
            run,
            450
          );
      }
    };

    void run();
  };

  const stopYOLO = () => {
    yoloRunningRef.current =
      false;

    if (
      yoloLoopRef.current !==
      null
    ) {
      window.clearTimeout(
        yoloLoopRef.current
      );

      yoloLoopRef.current =
        null;
    }
  };

  /*
   * START VIDEO RECORDING.
   */
  const startVideoRecording =
    (): boolean => {
      const video =
        sourceVideoRef.current ??
        videoRef.current;

      if (!video) {
        return false;
      }

      try {
        cameraCanvasRef.current =
          document.createElement(
            "canvas"
          );

        graphCanvasRef.current =
          document.createElement(
            "canvas"
          );

        cameraCanvasRef.current.width =
          640;
        cameraCanvasRef.current.height =
          720;
        graphCanvasRef.current.width =
          640;
        graphCanvasRef.current.height =
          720;

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

        const cameraStream =
          cameraCanvasRef.current.captureStream(
            RECORDING_FPS
          );

        const graphStream =
          graphCanvasRef.current.captureStream(
            RECORDING_FPS
          );

        const recorder =
          new MediaRecorder(
            cameraStream,
            {
              mimeType,
            }
          );

        const graphRecorder =
          new MediaRecorder(
            graphStream,
            {
              mimeType,
            }
          );

        cameraChunksRef.current =
          [];

        graphChunksRef.current =
          [];

        recorder.ondataavailable =
          (event) => {
            if (
              event.data.size >
              0
            ) {
              cameraChunksRef.current.push(
                event.data
              );
            }
          };

        graphRecorder.ondataavailable =
          (event) => {
            if (
              event.data.size >
              0
            ) {
              graphChunksRef.current.push(
                event.data
              );
            }
          };

        recorder.onstop = () => {
          const blob =
            new Blob(
              cameraChunksRef.current,
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

        graphRecorder.onstop =
          () => {
            recordedGraphRef.current =
              new Blob(
                graphChunksRef.current,
                {
                  type:
                    graphRecorder.mimeType ||
                    "video/webm",
                }
              );
          };

        recorder.start(1000);

        graphRecorder.start(1000);

        mediaRecorderRef.current =
          recorder;

        graphRecorderRef.current =
          graphRecorder;

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
      const stopRecorder = (
        recorder: MediaRecorder | null
      ) =>
        new Promise<void>(
          (resolve) => {
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

            recorder.stop();
          }
        );

      return new Promise(
        (resolve) => {
          const recorder =
            mediaRecorderRef.current;

          if (
            (!recorder ||
              recorder.state ===
                "inactive") &&
            (!graphRecorderRef.current ||
              graphRecorderRef.current
                .state ===
                "inactive")
          ) {
            resolve();
            return;
          }

          setVideoStatus(
            "PROCESSING"
          );

          stopRenderLoop();

          Promise.all([
            stopRecorder(
              recorder
            ),
            stopRecorder(
              graphRecorderRef.current
            ),
          ]).then(() => {
            window.setTimeout(
              () => {
                setVideoStatus(
                  "READY"
                );

                resolve();
              },
              0
            );
          });
        }
      );
    };

  /*
   * Stitch camera + CSI graph videos
   * side-by-side using pure browser APIs.
   * No ffmpeg needed.
   */
  const renderStitchedVideo =
    async (): Promise<Blob> => {
      if (
        !recordedVideoRef.current ||
        !recordedGraphRef.current
      ) {
        throw new Error(
          "Recorded camera or CSI graph video is missing"
        );
      }

      const HALF_W = 640;
      const FULL_H = 720;

      const canvas =
        document.createElement(
          "canvas"
        );

      canvas.width = HALF_W * 2;
      canvas.height = FULL_H;

      const ctx =
        canvas.getContext("2d")!;

      const makeVideo = (
        blob: Blob
      ): Promise<HTMLVideoElement> =>
        new Promise((resolve, reject) => {
          const video =
            document.createElement(
              "video"
            );

          video.muted = true;
          video.playsInline = true;
          video.preload = "auto";

          video.onloadeddata = () =>
            resolve(video);

          video.onerror = () =>
            reject(
              new Error(
                "Failed to load recorded video"
              )
            );

          video.src =
            URL.createObjectURL(blob);
        });

      const camVideo = await makeVideo(
        recordedVideoRef.current
      );

      const graphVideo = await makeVideo(
        recordedGraphRef.current
      );

      /*
       * Record the composite canvas
       * as a single video stream.
       */
      const stream =
        canvas.captureStream(30);

      const mimeType =
        MediaRecorder.isTypeSupported(
          "video/webm;codecs=vp9"
        )
          ? "video/webm;codecs=vp9"
          : "video/webm";

      const recorder =
        new MediaRecorder(stream, {
          mimeType,
          videoBitsPerSecond:
            4_000_000,
        });

      const chunks: Blob[] = [];

      recorder.ondataavailable =
        (event) => {
          if (event.data.size > 0) {
            chunks.push(event.data);
          }
        };

      const recorderDone =
        new Promise<Blob>(
          (resolve) => {
            recorder.onstop = () => {
              resolve(
                new Blob(chunks, {
                  type: mimeType,
                })
              );
            };
          }
        );

      recorder.start();

      /*
       * Play both videos in sync and
       * composite each frame onto the
       * canvas until the shorter one
       * ends.
       */
      await Promise.all([
        camVideo.play(),
        graphVideo.play(),
      ]);

      await new Promise<void>(
        (resolve) => {
          const draw = () => {
            if (
              camVideo.ended ||
              graphVideo.ended
            ) {
              recorder.stop();
              resolve();
              return;
            }

            ctx.fillStyle = "#000";
            ctx.fillRect(
              0,
              0,
              canvas.width,
              canvas.height
            );

            /*
             * Camera on the left half.
             */
            const camAspect =
              camVideo.videoWidth /
              (camVideo.videoHeight ||
                1);

            let camDrawW = HALF_W;
            let camDrawH =
              HALF_W / camAspect;

            if (camDrawH > FULL_H) {
              camDrawH = FULL_H;
              camDrawW =
                FULL_H * camAspect;
            }

            const camX =
              (HALF_W - camDrawW) / 2;

            const camY =
              (FULL_H - camDrawH) / 2;

            ctx.drawImage(
              camVideo,
              camX,
              camY,
              camDrawW,
              camDrawH
            );

            /*
             * CSI graph on the right
             * half.
             */
            ctx.drawImage(
              graphVideo,
              HALF_W,
              0,
              HALF_W,
              FULL_H
            );

            requestAnimationFrame(
              draw
            );
          };

          requestAnimationFrame(draw);
        }
      );

      /*
       * Clean up object URLs.
       */
      URL.revokeObjectURL(
        camVideo.src
      );

      URL.revokeObjectURL(
        graphVideo.src
      );

      return recorderDone;
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

    if (sourceVideoRef.current) {
      sourceVideoRef.current.pause();
      sourceVideoRef.current.srcObject =
        null;
      sourceVideoRef.current =
        null;
    }

    setCameraStatus("OFF");

    setCameraName(
      "No camera selected"
    );
  };

  /*
   * CSI sample handler.
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

    const latest =
      data[data.length - 1];

    if (!latest) {
      return;
    }

    if (latest.iValues?.length) {
      iDataRef.current =
        latest.iValues;

      setIData(latest.iValues);
    }

    if (latest.qValues?.length) {
      qDataRef.current =
        latest.qValues;

      setQData(latest.qValues);
    }

    if (!isRecordingRef.current) {
      return;
    }
  };

  /*
   * YOLO sample handler.
   */
  const handleYOLOData = (
    data: YOLOData
  ) => {
    currentYOLORef.current =
      data;

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
   * MODEL sample handler.
   */
  const handleModelData = (
    data: ModelData
  ) => {
    setModelHistory((previous) => [
      ...previous,
      data,
    ]);

    setCurrentModel(data);
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
      currentYOLORef.current =
        null;

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

    currentYOLORef.current =
      closest;

    setCurrentYOLO(closest);
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

  const stopSampleCSI = () => {
    sampleFramesRef.current = [];
    sampleIndexRef.current = 0;
  };

  const startSampleCSI =
    async () => {
      const frames =
        await loadSampleCSIFrames();

      sampleFramesRef.current =
        frames;

      sampleIndexRef.current = 0;

      setConnectionStatus(
        "CONNECTED"
      );
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

    recordedVideoRef.current =
      null;

    recordedGraphRef.current =
      null;

    iDataRef.current = [];
    qDataRef.current = [];
    currentYOLORef.current =
      null;

    setIsRecording(true);

    isRecordingRef.current =
      true;

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

    await startSampleCSI();

    startRenderLoop();

    if (yoloEnabled) {
      startYOLO();
    }
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

    stopYOLO();

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

    isRecordingRef.current =
      false;

    stopSampleCSI();

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
   * Download the stitched video generated
   * from the camera and CSI graph streams.
   * Uses pure canvas stitching (no ffmpeg).
   */
  const handleDownload =
    async () => {
      if (!lastSession) {
        return;
      }

      setVideoStatus(
        "PROCESSING"
      );

      const downloadBlob = (
        blob: Blob,
        filename: string
      ) => {
        const url =
          URL.createObjectURL(
            blob
          );

        const link =
          document.createElement(
            "a"
          );

        link.href = url;
        link.download = filename;

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

      try {
        const stitchedBlob =
          await renderStitchedVideo();

        setVideoStatus("READY");

        downloadBlob(
          stitchedBlob,
          `${lastSession.sessionId}.webm`
        );
      } catch (error) {
        console.warn(
          "[Wi-Track] Stitched export failed, falling back to separate files:",
          error
        );

        setVideoStatus("READY");

        if (
          recordedVideoRef.current
        ) {
          downloadBlob(
            recordedVideoRef.current,
            `${lastSession.sessionId}_camera.webm`
          );
        }

        if (
          recordedGraphRef.current
        ) {
          downloadBlob(
            recordedGraphRef.current,
            `${lastSession.sessionId}_csi.webm`
          );
        }
      }
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
              className={`control-button${
                modelEnabled
                  ? " model-active"
                  : ""
              }`}
              type="button"
              onClick={() => {
                setModelEnabled(
                  (prev) => {
                    const next = !prev;
                    modelEnabledRef.current = next;
                    if (!next && isRecordingRef.current) {
                      setCurrentModel(null);
                    }
                    return next;
                  }
                );
              }}
            >
              <span>◆</span>
              MODEL{" "}
              {modelEnabled
                ? "ON"
                : "OFF"}
            </button>

            <button
              className={`control-button${
                yoloEnabled
                  ? " yolo-active"
                  : ""
              }`}
              type="button"
              onClick={() => {
                setYoloEnabled(
                  (prev) => {
                    const next =
                      !prev;

                    if (
                      isRecordingRef.current
                    ) {
                      if (next) {
                        startYOLO();
                      } else {
                        stopYOLO();
                        currentYOLORef.current =
                          null;
                        setCurrentYOLO(
                          null
                        );
                      }
                    }

                    return next;
                  }
                );
              }}
            >
              <span>◇</span>
              YOLO{" "}
              {yoloEnabled
                ? "ON"
                : "OFF"}
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
