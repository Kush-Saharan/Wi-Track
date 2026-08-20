import time
import os
import pandas as pd
import numpy as np
from typing import Optional
from csi_parser import parse_csi_line, CSIPacket

class BaseReader:
    def read_packet() -> Optional[CSIPacket]:
        raise NotImplementedError
    def close(self):
        pass

class SerialReader(BaseReader):
    def __init__(self, port: str, baud_rate: int):
        import serial
        self.port = port
        self.baud_rate = baud_rate
        self.ser = serial.Serial(port, baud_rate, timeout=1)

    def read_packet() -> Optional[CSIPacket]:
        if not self.ser or not self.ser.is_open:
            return None
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        return parse_csi_line(line)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

class MockSerialReader(BaseReader):
    def __init__(self, csv_path: str, fps: int = 20):
        self.csv_path = csv_path
        self.interval = 1.0 / fps
        self.packets = []
        self.index = 0
        self._load_csv()

    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            return
        df = pd.read_csv(self.csv_path)
        cols_to_drop = ["PC_Timestamp_24h", "MAC_Address", "Activity_Label", "role", "mac", "rssi", "channel", "n_subcarriers"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        if len(df.columns) > 0 and df.dtypes.iloc[0] == object:
            df = df.iloc[:, 1:]
        
        vals = df.values.astype(float)
        for row in vals:
            if len(row) >= 128:
                raw = row[:128]
                i_vals = raw[0::2]
                q_vals = raw[1::2]
                amps = np.sqrt(i_vals**2 + q_vals**2)
                self.packets.append((i_vals.tolist(), q_vals.tolist(), amps.tolist()))

    def read_packet() -> Optional[CSIPacket]:
        if not self.packets:
            return None
        time.sleep(self.interval)
        i_vals, q_vals, amps = self.packets[self.index]
        self.index = (self.index + 1) % len(self.packets)
        return CSIPacket(
            i_values=i_vals,
            q_values=q_vals,
            amplitudes=amps,
            timestamp=time.time()
        )
