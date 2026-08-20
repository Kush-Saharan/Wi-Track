import os

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
BAUD_RATE = int(os.environ.get("BAUD_RATE", "115200"))

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "realtime_model.pkl"
)
CSI_SUBCARRIERS = 64
