# WiTrack v3 - Real-Time Live Monitor Setup

## 1. Where is the data coming from right now?
By default, the backend (`backend/main_realtime.py`) uses **mocked data**. Since your ESP32 isn't permanently plugged into the laptop, the backend automatically reads a loop of packets from `collected_data/final recordings/bathroom_group.csv`. This ensures the UI dashboard always has a live 20 FPS stream of amplitudes to display for presentation purposes, even without hardware connected.

## 2. Where is the older UI?
- **v1 (Initial Setup)**: Located in `frontend/src/v1/`.
- **v2 (Pre-Recorded Demo)**: The dual-channel synced demo (with video playback) is located in `frontend/src/v2/`.
To switch back to the older UI at any time, just edit `frontend/src/App.tsx` and change the import to point to `./v2/App` instead of `./v3/App`.

## 3. How to use Real ESP32 Hardware
To stop using the mock data and stream real CSI data directly from the ESP32 in real-time, follow these steps:

### A. Flash the ESP32 (Single Board Setup)
You only need **one** ESP32 board flashed as an Access Point (AP) to collect CSI data.
1. Use the firmware code located in `D:\WiTrack\active_ap`.
2. Connect your single ESP32 board via USB.
3. Flash the board using `idf.py build flash monitor`.
4. Your laptop/phone will connect to this ESP32 AP and send traffic (e.g., continuous pings) so the ESP32 can extract CSI from the incoming packets!

### B. Read from Serial Port
The ESP32 Receiver will dump raw CSI data over its Serial port. 
1. Open `backend/main_realtime.py`.
2. Delete the `MOCK DATA SOURCE` section.
3. Add the Python `pyserial` library to read from the COM port.
```python
import serial
ser = serial.Serial('COM3', 115200) # Replace COM3 with your ESP32's COM port

def read_real_esp32_packet():
    line = ser.readline().decode('utf-8').strip()
    # Parse the line (similar to how the CSI tool parses CSVs)
    # Return the 64-length amplitude numpy array
```
4. Replace the loop in the `/ws/data` websocket endpoint to await data from the serial port instead of ticking a `mock_index`.

## 4. The Model (train_realtime_model.py)
This folder contains the specific script (`train_realtime_model.py`) used to generate the live classification model.
- **What it does**: It combines all the CSV data you collected (Bathroom, Library, LibraryLift) into a universal "Empty vs Activity" Random Forest model.
- **Why**: Since you will be running this live in a room, you need a model that detects the universal signatures of human presence rather than overfitting to just the bathroom scenario.

## 5. Running the Demo
When ready to present the live v3 system:
1. Ensure your laptop webcam is enabled.
2. Run `.\start_realtimewitrack.ps1` from PowerShell.
3. Accept the Camera permission in the browser when the UI loads.
