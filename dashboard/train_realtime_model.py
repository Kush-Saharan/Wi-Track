import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "collected_data", "final recordings")
# Combine all data into empty vs activity
FILES = {
    "empty": ["bathroom_empty.csv", "library_empty.csv", "librarylift_empty.csv"],
    "activity": ["bathroom_single.csv", "bathroom_group.csv", 
                 "library_single.csv", "library_group.csv",
                 "librarylift_single.csv", "librarylift_group.csv"]
}

WINDOW = 100
STRIDE = 50
CROP_PACKETS = 3000  # crop 30 seconds

def load_amplitude(path):
    p = os.path.join(DATA_DIR, path)
    if not os.path.exists(p): return np.empty((0, 64))
    df = pd.read_csv(p)
    cols_to_drop = ["PC_Timestamp_24h", "MAC_Address", "Activity_Label", "role", "mac", "rssi", "channel", "n_subcarriers"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    if len(df.columns) > 0 and df.dtypes.iloc[0] == object:
        df = df.iloc[:, 1:]
    
    if len(df) > 2 * CROP_PACKETS:
        df = df.iloc[CROP_PACKETS : len(df) - CROP_PACKETS]
        
    X = df.values.astype(float)
    if X.shape[1] < 128: return np.empty((0, 64))
    return np.sqrt(X[:, 0::2] ** 2 + X[:, 1::2] ** 2)

empty_amps = []
for f in FILES["empty"]:
    a = load_amplitude(f)
    if len(a) > 0: empty_amps.append(a)
empty = np.vstack(empty_amps)

activity_amps = []
for f in FILES["activity"]:
    a = load_amplitude(f)
    if len(a) > 0: activity_amps.append(a)
activity = np.vstack(activity_amps)

# General Ruler based on all empty data
half = len(empty) // 2
ruler_src, empty_train = empty[:half], empty[half:]
avg = ruler_src.mean(0)
wob = ruler_src.std(0) + 1e-8

print(f"Empty data: {len(empty)} packets. Activity data: {len(activity)} packets.")

def cal(a): return (a - avg) / wob

sessions = [
    {"label": "no_activity", "amp": cal(empty_train)},
    {"label": "activity", "amp": cal(activity)}
]

def window_features(amp, W, S):
    idx = range(0, len(amp) - W + 1, S)
    out = np.empty((len(idx), amp.shape[1] * 4), dtype=np.float32)
    for r, w0 in enumerate(idx):
        w = amp[w0:w0 + W]
        mad = np.abs(np.diff(w, axis=0)).mean(0) if W > 1 else np.zeros(w.shape[1])
        out[r] = np.concatenate([w.std(0), mad, w.max(0) - w.min(0), w.mean(0)])
    return out

Fs, ys = [], []
for s in sessions:
    F = window_features(s["amp"], WINDOW, STRIDE)
    Fs.append(F)
    ys += [s["label"]] * len(F)

F = np.vstack(Fs)
y = np.array(ys)

Xtr, Xte, ytr, yte = train_test_split(F, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr)
acc = accuracy_score(yte, rf.predict(Xte))
print(f"Realtime Model Accuracy: {acc*100:.2f}%")

bundle = {
    "model": rf,
    "window_size": WINDOW,
    "avg": avg,
    "wob": wob
}
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "realtime_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"Saved {model_path}")
