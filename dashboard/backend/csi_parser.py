import time
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class CSIPacket:
    i_values: list[float]
    q_values: list[float]
    amplitudes: list[float]
    timestamp: float

def parse_csi_line(line: str) -> Optional[CSIPacket]:
    if not line or "CSI_DATA" not in line:
        return None

    try:
        if "[" in line and "]" in line:
            raw_array_str = line[line.index("[") + 1 : line.index("]")]
        else:
            parts = line.split(",")
            idx = 0
            for i, p in enumerate(parts):
                if "CSI_DATA" in p:
                    idx = i + 1
                    break
            raw_array_str = ",".join(parts[idx:])

        tokens = re.split(r"[\s,]+", raw_array_str.strip())
        numeric_vals = []
        for t in tokens:
            if not t:
                continue
            try:
                numeric_vals.append(float(t))
            except ValueError:
                continue

        if len(numeric_vals) < 2:
            return None

        raw = np.array(numeric_vals, dtype=float)
        i_vals = raw[0::2]
        q_vals = raw[1::2]
        
        min_len = min(len(i_vals), len(q_vals))
        i_vals = i_vals[:min_len]
        q_vals = q_vals[:min_len]

        amps = np.sqrt(i_vals**2 + q_vals**2)

        return CSIPacket(
            i_values=i_vals.tolist(),
            q_values=q_vals.tolist(),
            amplitudes=amps.tolist(),
            timestamp=time.time()
        )
    except Exception:
        return None
