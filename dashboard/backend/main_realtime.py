import asyncio
import pickle
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import serial

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use relative path to cleanly load from the top-level models folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "realtime_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    W = bundle["window_size"]
    avg = bundle["avg"]
    wob = bundle["wob"]
    print("Model loaded successfully")
except Exception as e:
    model, W, avg, wob = None, 100, 0, 1
    print(f"Failed to load model: {e}. Please run v3/train_realtime_model.py first!")

connected_clients = set()

# Initialize Serial Port - Update COM port as necessary
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

def get_serial_data(ser):
    """
    Reads a line from ESP32 serial, parses it to extract CSI amplitudes.
    Modify the parsing logic here to match your exact ESP32-CSI-Tool output format!
    """
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        # Example parsing: if the ESP32 prints comma separated CSI values:
        # "CSI_DATA,12,34,-5,..."
        if "CSI_DATA" in line:
            parts = line.split(',')
            # Adjust the slice depending on where your data actually starts
            data = np.array(parts[1:], dtype=float)
            
            # If data comes as I/Q pairs, calculate amplitude:
            # MAG = np.sqrt(data[0::2]**2 + data[1::2]**2)
            # return MAG
            
            # If data is already amplitude, just return it:
            return data
    except Exception as e:
        print(f"Serial read error: {e}")
    return None

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    
    # Open Serial connection when client connects
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        await websocket.send_json({"error": f"Failed to connect to {SERIAL_PORT}"})
        await websocket.close()
        return

    try:
        buffer = []
        while True:
            # We use an executor to prevent blocking the async loop with serial read
            amp = await asyncio.get_event_loop().run_in_executor(None, get_serial_data, ser)
            
            if amp is not None and len(amp) == 64:  # Assuming 64 subcarriers
                buffer.append(amp)
                if len(buffer) > W:
                    buffer.pop(0)
            
                if len(buffer) == W:
                    window = np.array(buffer)
                    cal = (window - avg) / wob
                    mad = np.abs(np.diff(cal, axis=0)).mean(0)
                    f = np.concatenate([cal.std(0), mad, cal.max(0) - cal.min(0), cal.mean(0)]).reshape(1, -1)
                    
                    pred = str(model.predict(f)[0]) if model else "unknown"
                    conf = float(model.predict_proba(f)[0].max()) if model else 0.0
                    motion = float(np.mean(np.abs(np.diff(window, axis=0))))
                    
                    payload = {
                        "amplitude": buffer[-1].tolist(),
                        "prediction": pred,
                        "confidence": conf,
                        "motion": motion,
                        "timestamp": time.time()
                    }
                    
                    await websocket.send_json(payload)
            
            # Small sleep to yield loop
            await asyncio.sleep(0.001)
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print("Client disconnected")
    finally:
        if ser and ser.is_open:
            ser.close()
