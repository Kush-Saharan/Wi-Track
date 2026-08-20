import asyncio
import csv
import itertools
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config
from csi_parser import CSIPacket
from serial_reader import SerialReader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MockCSIReader:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._rows = self._load_rows(csv_path)
        self._iterator = itertools.cycle(self._rows)

    def _load_rows(self, csv_path: str) -> list[tuple[list[float], list[float]]]:
        rows: list[tuple[list[float], list[float]]] = []

        with open(csv_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                i_values: list[float] = []
                q_values: list[float] = []

                index = 0
                while True:
                    i_raw = row.get(f"I{index}")
                    q_raw = row.get(f"Q{index}")

                    if i_raw is None or q_raw is None:
                        break

                    try:
                        i_values.append(float(i_raw))
                        q_values.append(float(q_raw))
                    except ValueError:
                        i_values.append(0.0)
                        q_values.append(0.0)

                    index += 1

                if i_values and q_values:
                    rows.append((i_values, q_values))

        if not rows:
            raise RuntimeError(f"No CSI rows found in mock CSV: {csv_path}")

        return rows

    def read_packet(self) -> CSIPacket:
        i_values, q_values = next(self._iterator)
        amplitudes = [
            (i_value**2 + q_value**2) ** 0.5
            for i_value, q_value in zip(i_values, q_values)
        ]

        return CSIPacket(
            i_values=i_values,
            q_values=q_values,
            amplitudes=amplitudes,
            timestamp=time.time(),
        )

    def close(self):
        return None

def create_reader():
    if config.DATA_SOURCE == "serial":
        reader = SerialReader(config.SERIAL_PORT, config.BAUD_RATE)
        print(
            f"[Backend] Connected to serial port {config.SERIAL_PORT} "
            f"@ {config.BAUD_RATE}"
        )
        return reader

    reader = MockCSIReader(config.MOCK_CSI_CSV)
    print(f"[Backend] Streaming mock CSI data from {config.MOCK_CSI_CSV}")
    return reader

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    reader = None

    try:
        reader = create_reader()
    except Exception as e:
        print(f"[Backend] Could not initialize {config.DATA_SOURCE} source: {e}")
        await websocket.send_json({"error": f"Failed to initialize {config.DATA_SOURCE} source"})
        await websocket.close()
        return

    try:
        loop = asyncio.get_event_loop()
        while True:
            packet = await loop.run_in_executor(None, reader.read_packet)
            
            if packet is not None:
                csi_payload = {
                    "type": "csi",
                    "timestamp": packet.timestamp,
                    "i": packet.i_values,
                    "q": packet.q_values,
                    "iValues": packet.i_values,
                    "qValues": packet.q_values,
                    "amplitude": packet.amplitudes
                }
                await websocket.send_json(csi_payload)

                # Model plugin

            if config.DATA_SOURCE != "serial":
                await asyncio.sleep(1 / config.MOCK_FPS)
            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        print("[Backend] WebSocket disconnected")
    except Exception as e:
        print(f"[Backend] WebSocket error: {e}")
    finally:
        if reader:
            reader.close()
