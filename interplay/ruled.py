"""Step 5: ruled surfaces between pairs of players.

At each beat a line (ruling) joins two players in the screen space; the rulings in time order
weave a surface. Twist is the angle the ruling turns over one bar: two players moving alike keep
it parallel (a flat surface); one moving and the other following late turns it.
"""

import itertools

import numpy as np
import pandas as pd

from .config import BEATS_PER_BAR, DATA_DIR, INSTRUMENTS
from .screen_space import positions, trajectories

PAIRS = list(itertools.combinations(INSTRUMENTS, 2))


def twist(ruling: np.ndarray) -> np.ndarray:
    """Degrees the ruling turns between each beat and the beat one bar later."""
    a, b = ruling[:-BEATS_PER_BAR], ruling[BEATS_PER_BAR:]
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide((a * b).sum(axis=1), norm, out=np.full(len(norm), np.nan), where=norm > 0)
    return np.r_[np.degrees(np.arccos(np.clip(cos, -1, 1))), np.full(BEATS_PER_BAR, np.nan)]


def with_twist(table: pd.DataFrame) -> pd.DataFrame:
    for x, y in PAIRS:
        table[f"{x}_{y}_twist"] = twist(positions(table, y) - positions(table, x))
    return table


def run() -> None:
    table = with_twist(trajectories())
    table.to_csv(DATA_DIR / "ruled.csv", index_label="beat")
    # A bar counts for a section only if it ends in the same section (not in a transition).
    within = table.section.eq(table.section.shift(-BEATS_PER_BAR))
    summary = table[within].groupby("section")[[f"{x}_{y}_twist" for x, y in PAIRS]].mean()
    print("Twist (degrees per bar) by section:")
    print(summary.round(1).to_string())
