"""Step 4: each player as a point in the screen space (frequency, loudness, density).

Grains are smoothed with a Gaussian window (sigma 2 beats). Density is grains per beat; frequency
and loudness are the window-weighted means, pulled to the player's mean when the player is
silent. Each axis is z-scored per player, so the origin is that player's usual state.
"""

import numpy as np
import pandas as pd

from .beats import load_beats
from .config import DATA_DIR, INSTRUMENTS, section_of
from .grains import FREQUENCY, load_grains

AXES = ("frequency", "intensity", "density")
SIGMA_BEATS = 2.0
PRIOR_WEIGHT = 0.5


def features(
    grains: pd.DataFrame, freq: str, times: np.ndarray, period: float, chunk: int = 2048
) -> pd.DataFrame:
    """Smoothed frequency, intensity and density at arbitrary times.

    Every grain counts for density and intensity; frequency uses only grains with a value (bass
    notes where pYIN found no pitch have none).
    """
    sigma = SIGMA_BEATS * period
    onsets = grains.onset.to_numpy()
    values = {"frequency": grains[freq].to_numpy(), "intensity": grains.intensity.to_numpy()}
    parts = []
    for start in range(0, len(times), chunk):
        t = times[start : start + chunk]
        w = np.exp(-0.5 * ((t[:, None] - onsets[None, :]) / sigma) ** 2)
        part = {"density": w.sum(axis=1) * period / (sigma * np.sqrt(2 * np.pi))}
        for axis, v in values.items():
            known = np.isfinite(v)
            wk, vk = w[:, known], v[known]
            part[axis] = (wk @ vk + PRIOR_WEIGHT * vk.mean()) / (wk.sum(axis=1) + PRIOR_WEIGHT)
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)[list(AXES)]


def positions_at(times: np.ndarray) -> dict[str, np.ndarray]:
    """Positions at arbitrary times, z-scored with the per-beat mean and std."""
    beats = load_beats()
    period = float(np.median(beats.end - beats.start))
    grains = load_grains()
    out = {}
    for name in INSTRUMENTS:
        reference = features(grains[name], FREQUENCY[name], beats.start.to_numpy(), period)
        raw = features(grains[name], FREQUENCY[name], times, period)
        out[name] = ((raw - reference.mean()) / reference.std(ddof=0)).to_numpy()
    return out


def trajectories() -> pd.DataFrame:
    """One row per beat: time, section and every player's position."""
    beats = load_beats()
    times = beats.start.to_numpy()
    table = pd.DataFrame(
        {
            "time": times,
            "section": [section_of(t, tr) for t, tr in zip(times, beats.transition, strict=True)],
        }
    )
    for name, p in positions_at(times).items():
        for k, axis in enumerate(AXES):
            table[f"{name}_{axis}"] = p[:, k]
    return table


def positions(table: pd.DataFrame, name: str) -> np.ndarray:
    return table[[f"{name}_{a}" for a in AXES]].to_numpy()


def run() -> None:
    trajectories().to_csv(DATA_DIR / "screen_space.csv", index_label="beat")
