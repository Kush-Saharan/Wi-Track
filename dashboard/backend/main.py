import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config
from serial_reader import SerialReader, MockSerialReader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_reader():
    if config.USE_MOCK:
        print(f"[Backend] Starting MockSerialReader using CSV: {config.MOCK_CSV_PATH}")
        return MockSerialReader(config.MOCK_CSV_PATH)
    else:
        print(f"[Backend] Starting SerialReader on port {config.SERIAL_PORT} @ {config.BAUD_RATE}")
        return SerialReader(config.SERIAL_PORT, config.BAUD_RATE)

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    reader = None

    try:
        reader = create_reader()
    except Exception as e:
        await websocket.send_json({"error": f"Failed to initialize serial reader: {e}"})
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

                # ====================================================
                # TEAMMATE MODEL PLUGIN PLACEHOLDER
                # Your teammate can add model prediction here:
                # pred, conf = model.predict(packet.amplitudes)
                # await websocket.send_json({
                #     "type": "model",
                #     "prediction": pred,
                #     "confidence": conf,
                #     "timestamp": packet.timestamp
                # })
                # ====================================================

            await asyncio.sleep(0.001)

    except WebSocketDisconnect:
        print("[Backend] WebSocket disconnected")
    except Exception as e:
        print(f"[Backend] WebSocket error: {e}")
    finally:
        if reader:
            reader.close()
