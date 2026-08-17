export interface CSIData {
  timestamp: number;
  i: number;
  q: number;
  iValues?: number[];
  qValues?: number[];
}

export interface YOLODetection {
  label: string;
  confidence: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export interface YOLOData {
  timestamp: number;
  detections: YOLODetection[];
}

export interface ModelData {
  timestamp: number;
  prediction: string;
  confidence?: number;
  status?: string;
}

type CSIMessageHandler = (data: CSIData[]) => void;
type YOLOMessageHandler = (data: YOLOData) => void;
type ModelMessageHandler = (data: ModelData) => void;
type ConnectionHandler = () => void;
type ErrorHandler = (error: Event) => void;

interface BackendMessage {
  type?: string;
  timestamp?: number;
  data?: unknown;
  payload?: unknown;
  frames?: unknown;
  samples?: unknown;
  detections?: unknown;
  prediction?: unknown;
  confidence?: unknown;
  status?: unknown;
  i?: unknown;
  q?: unknown;
  iValues?: unknown;
  qValues?: unknown;
}

function toTimestamp(value: unknown): number {
  const timestamp = Number(value);

  if (Number.isFinite(timestamp)) {
    // Accept seconds as well as milliseconds.
    return timestamp < 1e12
      ? timestamp * 1000
      : timestamp;
  }

  return Date.now();
}

function toNumber(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function toNumberArray(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }

  const values = value
    .map((item) => Number(item))
    .filter(Number.isFinite);

  return values.length > 0 ? values : undefined;
}

function average(values: number[] | undefined): number {
  if (!values || values.length === 0) {
    return 0;
  }

  return (
    values.reduce((sum, value) => sum + value, 0) /
    values.length
  );
}

function unwrap(value: unknown): unknown {
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;

    if (object.data !== undefined) {
      return object.data;
    }

    if (object.payload !== undefined) {
      return object.payload;
    }
  }

  return value;
}

function normalizeCSI(value: unknown): CSIData[] {
  const unwrapped = unwrap(value);

  const items = Array.isArray(unwrapped)
    ? unwrapped
    : unwrapped && typeof unwrapped === "object"
      ? (() => {
          const object = unwrapped as BackendMessage;
          if (Array.isArray(object.frames)) return object.frames;
          if (Array.isArray(object.samples)) return object.samples;
          return [unwrapped];
        })()
      : [];

  return items.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }

    const object = item as BackendMessage;

    const iValues =
      toNumberArray(object.iValues) ??
      toNumberArray(object.i);

    const qValues =
      toNumberArray(object.qValues) ??
      toNumberArray(object.q);

    const i =
      toNumber(object.i) ??
      average(iValues);

    const q =
      toNumber(object.q) ??
      average(qValues);

    return [
      {
        timestamp: toTimestamp(object.timestamp),
        i,
        q,
        iValues,
        qValues,
      },
    ];
  });
}

function normalizeYOLO(value: unknown): YOLOData | null {
  const unwrapped = unwrap(value);

  if (!unwrapped || typeof unwrapped !== "object") {
    return null;
  }

  const object = unwrapped as BackendMessage;
  const rawDetections = Array.isArray(object.detections)
    ? object.detections
    : [];

  const detections: YOLODetection[] = rawDetections.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }

    const detection = item as Record<string, unknown>;
    const label = String(
      detection.label ??
        detection.className ??
        detection.class ??
        "object"
    );
    const confidence = toNumber(
      detection.confidence ?? detection.score
    ) ?? 0;

    return [
      {
        label,
        confidence,
        x: toNumber(detection.x),
        y: toNumber(detection.y),
        width: toNumber(detection.width),
        height: toNumber(detection.height),
      },
    ];
  });

  return {
    timestamp: toTimestamp(object.timestamp),
    detections,
  };
}

function normalizeModel(value: unknown): ModelData | null {
  const unwrapped = unwrap(value);

  if (!unwrapped || typeof unwrapped !== "object") {
    return null;
  }

  const object = unwrapped as BackendMessage;
  const prediction =
    object.prediction ??
    (object as Record<string, unknown>).label ??
    (object as Record<string, unknown>).className;

  if (prediction === undefined) {
    return null;
  }

  return {
    timestamp: toTimestamp(object.timestamp),
    prediction: String(prediction),
    confidence: toNumber(object.confidence),
    status:
      object.status !== undefined
        ? String(object.status)
        : undefined,
  };
}

export class CSIWebSocket {
  private socket: WebSocket | null = null;
  private readonly url: string;
  private readonly onCSIData: CSIMessageHandler;
  private readonly onYOLOData: YOLOMessageHandler;
  private readonly onModelData: ModelMessageHandler;
  private readonly onOpen: ConnectionHandler;
  private readonly onClose: ConnectionHandler;
  private readonly onError: ErrorHandler;

  constructor(
    url: string,
    onCSIData: CSIMessageHandler,
    onYOLOData: YOLOMessageHandler,
    onModelData: ModelMessageHandler,
    onOpen: ConnectionHandler,
    onClose: ConnectionHandler,
    onError: ErrorHandler
  ) {
    this.url = url;
    this.onCSIData = onCSIData;
    this.onYOLOData = onYOLOData;
    this.onModelData = onModelData;
    this.onOpen = onOpen;
    this.onClose = onClose;
    this.onError = onError;
  }

  connect() {
    this.disconnect();

    try {
      this.socket = new WebSocket(this.url);

      this.socket.onopen = () => {
        console.log("[Wi-Track] WebSocket connected");
        this.onOpen();
      };

      this.socket.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.socket.onerror = (event) => {
        console.error("[Wi-Track] WebSocket error:", event);
        this.onError(event);
      };

      this.socket.onclose = () => {
        console.log("[Wi-Track] WebSocket disconnected");
        this.onClose();
        this.socket = null;
      };
    } catch (error) {
      console.error("[Wi-Track] WebSocket setup error:", error);
      this.onError(error as Event);
    }
  }

  disconnect() {
    if (!this.socket) {
      return;
    }

    this.socket.onopen = null;
    this.socket.onmessage = null;
    this.socket.onerror = null;
    this.socket.onclose = null;

    if (
      this.socket.readyState === WebSocket.OPEN ||
      this.socket.readyState === WebSocket.CONNECTING
    ) {
      this.socket.close();
    }

    this.socket = null;
  }

  private handleMessage(rawMessage: unknown) {
    let message: unknown = rawMessage;

    if (typeof rawMessage === "string") {
      try {
        message = JSON.parse(rawMessage);
      } catch {
        console.warn("[Wi-Track] Ignoring non-JSON WebSocket message");
        return;
      }
    }

    if (!message || typeof message !== "object") {
      return;
    }

    const object = message as BackendMessage;
    const type = String(object.type ?? "csi").toLowerCase();

    if (
      type.includes("yolo") ||
      type.includes("detect")
    ) {
      const data = normalizeYOLO(message);
      if (data) {
        this.onYOLOData(data);
      }
      return;
    }

    if (
      type.includes("model") ||
      type.includes("prediction") ||
      type.includes("predict")
    ) {
      const data = normalizeModel(message);
      if (data) {
        this.onModelData(data);
      }
      return;
    }

    const data = normalizeCSI(message);
    if (data.length > 0) {
      this.onCSIData(data);
    }
  }
}