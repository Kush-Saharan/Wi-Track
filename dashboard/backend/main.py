import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config
from serial_reader import SerialReader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    reader = None

    try:
        reader = SerialReader(config.SERIAL_PORT, config.BAUD_RATE)
        print(f"[Backend] Connected to serial port {config.SERIAL_PORT} @ {config.BAUD_RATE}")
    except Exception as e:
        print(f"[Backend] Could not open serial port {config.SERIAL_PORT}: {e}")
        await websocket.send_json({"error": f"Failed to connect to {config.SERIAL_PORT}"})
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

            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        print("[Backend] WebSocket disconnected")
    except Exception as e:
        print(f"[Backend] WebSocket error: {e}")
    finally:
        if reader:
            reader.close()
