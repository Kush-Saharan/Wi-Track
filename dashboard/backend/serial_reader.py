from typing import Optional
from csi_parser import parse_csi_line, CSIPacket

class SerialReader:
    def __init__(self, port: str, baud_rate: int):
        import serial
        self.port = port
        self.baud_rate = baud_rate
        self.ser = serial.Serial(port, baud_rate, timeout=1)

    def read_packet(self) -> Optional[CSIPacket]:
        if not self.ser or not self.ser.is_open:
            return None
        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        return parse_csi_line(line)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
