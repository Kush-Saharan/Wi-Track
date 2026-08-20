import os

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
BAUD_RATE = int(os.environ.get("BAUD_RATE", "115200"))
DATA_SOURCE = os.environ.get("DATA_SOURCE", "mock").lower()
MOCK_FPS = float(os.environ.get("MOCK_FPS", "20"))
MOCK_CSI_CSV = os.environ.get(
    "MOCK_CSI_CSV",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "frontend",
        "src",
        "piyush",
        "csi_data.csv",
    ),
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "realtime_model.pkl"
)
CSI_SUBCARRIERS = 64
