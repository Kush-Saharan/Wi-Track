import os

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
BAUD_RATE = int(os.environ.get("BAUD_RATE", "115200"))
USE_MOCK = os.environ.get("USE_MOCK", "false").lower() in ("true", "1", "t")

DEFAULT_MOCK_CSV = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "src", "vijay", "csi_data.csv"
)
MOCK_CSV_PATH = os.environ.get("MOCK_CSV_PATH", DEFAULT_MOCK_CSV)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "realtime_model.pkl"
)
CSI_SUBCARRIERS = 64
