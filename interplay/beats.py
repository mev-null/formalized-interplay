"""Step 3 (with `grains`): the beat grid, estimated part by part (`config.PARTS`).

Each part is tracked on its own, then each beat is shifted by the local circular-mean offset of
the onsets of the instruments that keep the pulse in that part. Beats near a part boundary are
marked as transitions: there the pulse is unsteady even to a listener.
"""

import librosa
import numpy as np
import pandas as pd

from .config import DATA_DIR, HOP, PARTS, SR, TEMPO_PRIOR, TRANSITION

CHORD_GAP = 0.03
# Spread of the tempo prior around TEMPO_PRIOR, in octaves: keeps short parts from locking
# onto 2/3 or 1/2 tempo.
TEMPO_SPREAD = 0.1


def reference_onsets(
    grains: dict[str, pd.DataFrame], names: tuple[str, ...], start: float, end: float
) -> np.ndarray:
    """Onsets of the reference instruments in a part, with chord notes merged into one."""
    t = np.sort(np.concatenate([grains[n].onset.to_numpy() for n in names]))
    t = t[np.r_[True, np.diff(t) > CHORD_GAP]]
    return t[(t >= start) & (t < end)]


def track(mix: np.ndarray, start: float, end: float) -> np.ndarray:
    y = mix[int(start * SR) : int(end * SR)]
    tempo = librosa.feature.tempo(
        y=y, sr=SR, hop_length=HOP, start_bpm=TEMPO_PRIOR, std_bpm=TEMPO_SPREAD
    )[0]
    _, times = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, units="time", bpm=tempo)
    return times + start


def align(
    starts: np.ndarray, onsets: np.ndarray, half_window: int = 8, min_onsets: int = 6
) -> np.ndarray:
    """Shift each beat by the circular-mean offset of the reference onsets nearby.

    Onsets within ±half_window beats count; with fewer than min_onsets the offset is interpolated.
    """
    if len(starts) < 2 or len(onsets) == 0:
        return starts
    period = np.median(np.diff(starts))
    nearest = np.abs(onsets[:, None] - starts[None, :]).argmin(axis=1)
    rotor = np.exp(2j * np.pi * (onsets - starts[nearest]) / period)
    offsets = np.full(len(starts), np.nan)
    for i in range(len(starts)):
        near = np.abs(nearest - i) <= half_window
        if near.sum() >= min_onsets:
            offsets[i] = np.angle(rotor[near].mean()) / (2 * np.pi) * period
    shifted = starts + pd.Series(offsets).interpolate(limit_direction="both").fillna(0).to_numpy()
    return shifted[np.r_[True, np.diff(shifted) > 0]]


def fill_gaps(starts: np.ndarray, parts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evenly fill gaps of more than 1.5 beats (the tracker can miss a part's edges)."""
    period = np.median(np.diff(starts))
    out_t, out_p = [starts[0]], [parts[0]]
    for t, p in zip(starts[1:], parts[1:], strict=True):
        n = int(round((t - out_t[-1]) / period))
        if n > 1 and t - out_t[-1] > 1.5 * period:
            out_t += list(np.linspace(out_t[-1], t, n + 1)[1:-1])
            out_p += [p] * (n - 1)
        out_t.append(t)
        out_p.append(p)
    return np.array(out_t), np.array(out_p)


def part_bounds(duration: float):
    for name, (start, end, refs) in PARTS.items():
        yield name, start, duration if end is None else end, refs


def estimate(mix: np.ndarray, grains: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, np.ndarray]:
    """Aligned beats (start, end, part, transition) and the raw tracked beats before alignment."""
    duration = len(mix) / SR
    rows, raw = [], []
    for name, start, end, refs in part_bounds(duration):
        tracked = track(mix, start, end)
        raw.append(tracked)
        shifted = align(tracked, reference_onsets(grains, refs, start, end))
        rows += [(t, name) for t in shifted if start <= t < end]
    starts = np.array([t for t, _ in rows])
    parts = np.array([p for _, p in rows])
    half = np.median(np.diff(starts)) / 2
    keep = [0]
    for i in range(1, len(starts)):
        if starts[i] - starts[keep[-1]] > half:
            keep.append(i)
    starts, parts = fill_gaps(starts[keep], parts[keep])
    boundaries = np.array([start for _, start, _, _ in part_bounds(duration)][1:])
    beats = pd.DataFrame(
        {
            "start": starts,
            "end": np.r_[starts[1:], duration],
            "part": parts,
            "transition": np.abs(starts[:, None] - boundaries[None, :]).min(axis=1) < TRANSITION,
        }
    )
    return beats, np.concatenate(raw)


def load_beats() -> pd.DataFrame:
    """The beat grid, without the final beat that spans the held last chord."""
    beats = pd.read_csv(DATA_DIR / "beats.csv", index_col="beat")
    duration = beats.end - beats.start
    return beats[duration < 2 * duration.median()]
