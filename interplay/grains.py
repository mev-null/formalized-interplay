"""Step 3: grains and beats.

A grain is one sound event (a note or a hit) with its onset time, a frequency value (pitch, or
brightness for the drums) and an intensity. The beat grid (`beats.py`) gives the time steps of
the screen space.
"""

import librosa
import numpy as np
import pandas as pd

from .beats import estimate
from .config import DATA_DIR, HOP, INSTRUMENTS, SR, STEM_DIR
from .transcribe import NOTES

GRAIN_WINDOW = 0.12
# Quiet brush hits need a lower threshold than librosa's default 0.07.
DRUM_DELTA = 0.02
FREQUENCY = {"piano": "pitch", "bass": "pitch", "drums": "brightness"}
# Bass that leaks into the piano transcription: low notes, or very soft notes, sounding with a
# bass onset.
LEAK_PITCH = 40
LEAK_WINDOW = 0.05
LEAK_VELOCITY = 20
LEAK_SOFT_WINDOW = 0.08
# Bass that leaks into the drum stem has a far lower zero-crossing rate than any hit.
DRUM_MIN_BRIGHTNESS = 0.05


def load(name: str) -> np.ndarray:
    y, _ = librosa.load(STEM_DIR / f"{name}.wav", sr=SR, mono=True)
    return y


def onsets(y: np.ndarray, delta: float = 0.07) -> np.ndarray:
    return librosa.onset.onset_detect(
        y=y, sr=SR, hop_length=HOP, units="time", backtrack=True, delta=delta
    )


def window_frames(times: np.ndarray) -> tuple[np.ndarray, int]:
    return librosa.time_to_frames(times, sr=SR, hop_length=HOP), max(
        1, int(GRAIN_WINDOW * SR / HOP)
    )


def peak_db(y: np.ndarray, times: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    frames, span = window_frames(times)
    return np.array([librosa.amplitude_to_db(rms[f : f + span].max(), ref=1.0) for f in frames])


def bass_grains(y: np.ndarray) -> pd.DataFrame:
    t = onsets(y)
    f0, voiced, _ = librosa.pyin(y, fmin=35.0, fmax=400.0, sr=SR, hop_length=HOP)
    midi = np.where(voiced, librosa.hz_to_midi(f0), np.nan)
    frames, span = window_frames(t)
    pitch = [
        np.nanmedian(midi[f : f + span]) if np.isfinite(midi[f : f + span]).any() else np.nan
        for f in frames
    ]
    return pd.DataFrame({"onset": t, "pitch": pitch, "intensity": peak_db(y, t)})


def drum_grains(y: np.ndarray) -> pd.DataFrame:
    t = onsets(y, DRUM_DELTA)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]
    frames, span = window_frames(t)
    return pd.DataFrame(
        {
            "onset": t,
            "brightness": [zcr[f : f + span].mean() for f in frames],
            "intensity": peak_db(y, t),
        }
    )


def piano_grains() -> pd.DataFrame:
    notes = pd.read_csv(NOTES)
    return notes.rename(columns={"velocity": "intensity"})[
        ["onset", "pitch", "intensity", "offset"]
    ]


def run() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    stems = {name: load(name) for name in INSTRUMENTS}
    grains = {
        "piano": piano_grains(),
        "bass": bass_grains(stems["bass"]),
        "drums": drum_grains(stems["drums"]),
    }
    for name, table in grains.items():
        table.to_csv(DATA_DIR / f"grains_{name}.csv", index=False)
        print(f"{name}: {len(table)} grains")
    beats, _ = estimate(sum(stems.values()), load_grains())
    beats.to_csv(DATA_DIR / "beats.csv", index_label="beat")
    tempo = beats.groupby("part", sort=False).apply(
        lambda b: 60 / np.median(np.diff(b.start)), include_groups=False
    )
    print(f"{len(beats)} beats ({beats.transition.sum()} in transitions); median tempo by part:")
    print(tempo.round(1).to_string())


def load_grains() -> dict[str, pd.DataFrame]:
    """Grains with leakage between stems removed."""
    grains = {name: pd.read_csv(DATA_DIR / f"grains_{name}.csv") for name in INSTRUMENTS}
    piano, bass = grains["piano"], grains["bass"]
    gap = np.abs(piano.onset.values[:, None] - bass.onset.values[None, :]).min(axis=1)
    low = (piano.pitch < LEAK_PITCH) & (gap < LEAK_WINDOW)
    soft = (piano.intensity < LEAK_VELOCITY) & (gap < LEAK_SOFT_WINDOW)
    grains["piano"] = piano[~(low | soft)].reset_index(drop=True)
    drums = grains["drums"]
    grains["drums"] = drums[drums.brightness >= DRUM_MIN_BRIGHTNESS].reset_index(drop=True)
    return grains
