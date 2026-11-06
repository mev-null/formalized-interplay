"""Step 6: the report's figures (§3.4), the middle 16 bars of each section.

Figure 1: the three players' paths in the screen space.
Figure 2: the ruled surface of each pair, drawn as a sheet.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import BEATS_PER_BAR, DATA_DIR, FIG_DIR, INSTRUMENTS, SECTIONS, TITLES, section_of
from .ruled import PAIRS
from .screen_space import positions, positions_at

COLORS = {"piano": "#3b6fb6", "bass": "#c8553d", "drums": "#5a9e6f"}
PAIR_COLORS = {
    ("piano", "bass"): "#7d4fb8",
    ("piano", "drums"): "#2a9d9d",
    ("bass", "drums"): "#b8860b",
}
LIMIT = 2.6
SHOWN_BARS = 16
RATE = 30


def pair_label(pair: tuple[str, str]) -> str:
    return "–".join(p.capitalize() for p in pair)


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def shown_span(table: pd.DataFrame, section: str) -> tuple[float, float]:
    """Start and end of the middle SHOWN_BARS bars of a section (equal spans in every panel)."""
    bars = table.iloc[::BEATS_PER_BAR]
    idx = np.flatnonzero([section_of(t) == section for t in bars.time])
    first = idx[0] + (len(idx) - SHOWN_BARS) // 2
    return bars.time.iat[first], bars.time.iat[first + SHOWN_BARS]


def style_axes(ax, fontsize: int = 7, labelpad: int = -10) -> None:
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-LIMIT, LIMIT)
    ax.set_xlabel("Register", fontsize=fontsize, labelpad=labelpad)
    ax.set_ylabel("Loudness", fontsize=fontsize, labelpad=labelpad)
    ax.set_zlabel("Density", fontsize=fontsize, labelpad=labelpad)
    ax.tick_params(labelsize=0, pad=-4)
    ax.view_init(elev=18, azim=-60)


def fig_trajectories(table: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9, 4.4))
    for col, section in enumerate(SECTIONS):
        ax = fig.add_subplot(1, len(SECTIONS), col + 1, projection="3d")
        ax.computed_zorder = False
        start, end = shown_span(table, section)
        s = table[(table.time >= start) & (table.time < end)]
        p = {name: np.clip(positions(s, name), -LIMIT, LIMIT) for name in INSTRUMENTS}
        for x, y in PAIRS:
            for a, b in zip(p[x], p[y], strict=True):
                ax.plot(
                    *np.stack([a, b]).T, color=PAIR_COLORS[(x, y)], lw=0.5, alpha=0.18, zorder=1
                )
        for name in INSTRUMENTS:
            ax.plot(*p[name].T, color=COLORS[name], lw=1.2, zorder=3)
        style_axes(ax)
        ax.set_title(f"{TITLES[section]}\n{fmt(start)}–{fmt(end)} ({SHOWN_BARS} bars)", fontsize=10)
    handles = [
        plt.Line2D([], [], color=COLORS[n], lw=1.4, label=n.capitalize()) for n in INSTRUMENTS
    ]
    handles += [
        plt.Line2D([], [], color=PAIR_COLORS[p], lw=1.4, label=f"Rulings {pair_label(p)}")
        for p in PAIRS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.06),
    )
    fig.savefig(FIG_DIR / "fig1_trajectories.png", dpi=200, bbox_inches="tight")


def draw_sheet(ax, a: np.ndarray, b: np.ndarray, color: str) -> None:
    """Quads between consecutive rulings, light early to dark late, with a ruling every beat."""
    quads = np.stack([a[:-1], a[1:], b[1:], b[:-1]], axis=1)
    rgb = np.array(to_rgb(color))
    t = np.linspace(0, 1, len(quads))[:, None]
    faces = np.column_stack([(1 - t) * (0.55 + 0.45 * rgb) + t * rgb, np.full(len(quads), 0.85)])
    ax.add_collection3d(
        Poly3DCollection(
            quads,
            facecolors=faces,
            edgecolors=faces,
            linewidths=0.2,
            shade=True,
            lightsource=LightSource(azdeg=225, altdeg=45),
        )
    )
    step = max(len(a) // (SHOWN_BARS * BEATS_PER_BAR), 1)
    for k in range(0, len(a), step):
        ax.plot(*np.stack([a[k], b[k]]).T, color="white", lw=0.3, alpha=0.35)


def fig_ruled_surfaces(table: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 7.6))
    for row, section in enumerate(SECTIONS):
        start, end = shown_span(table, section)
        fine = {
            name: np.clip(p, -LIMIT, LIMIT)
            for name, p in positions_at(np.arange(start, end, 1 / RATE)).items()
        }
        shown = table[(table.time >= start) & (table.time < end)]
        for col, (x, y) in enumerate(PAIRS):
            ax = fig.add_subplot(
                len(SECTIONS), len(PAIRS), row * len(PAIRS) + col + 1, projection="3d"
            )
            ax.computed_zorder = False
            draw_sheet(ax, fine[x], fine[y], PAIR_COLORS[(x, y)])
            for name in (x, y):
                ax.plot(*fine[name].T, color=COLORS[name], lw=0.7, alpha=0.8)
            style_axes(ax, fontsize=6, labelpad=-12)
            ax.set_title(
                f"{pair_label((x, y))}: {shown[f'{x}_{y}_twist'].mean():.0f}°/bar", fontsize=9
            )
            if col == 0:
                ax.text2D(
                    -0.08,
                    0.5,
                    f"{TITLES[section]}\n{fmt(start)}–{fmt(end)}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
    handles = [
        plt.Line2D([], [], color=COLORS[n], lw=1.2, label=n.capitalize()) for n in INSTRUMENTS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.06),
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.12)
    fig.savefig(FIG_DIR / "fig2_ruled_surfaces.png", dpi=200, bbox_inches="tight")


def run() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    table = pd.read_csv(DATA_DIR / "ruled.csv", index_col="beat")
    fig_trajectories(table)
    fig_ruled_surfaces(table)
