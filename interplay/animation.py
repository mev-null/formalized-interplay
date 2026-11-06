"""Animation of the ruled surfaces over the whole track (not in the report; shown in the README).

The GIF has no sound. `--realtime` writes an MP4 synced to the recording and muxes in the local
audio for private listening; that file must never be published.
"""

import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .config import AUDIO, DATA_DIR, FIG_DIR, INSTRUMENTS, SECTIONS, TITLES, WORK_DIR
from .ruled import PAIRS
from .screen_space import AXES, positions

BACKGROUND = "#000000"
FOREGROUND = "#e6e6e6"
MUTED = "#8a8a8a"
GRID = "#2a2a2a"
COLORS = {"piano": "#6fa8ff", "bass": "#ff7a5c", "drums": "#6fd08c"}
PAIR_COLORS = {
    ("piano", "bass"): "#c58cff",
    ("piano", "drums"): "#5fe0e0",
    ("bass", "drums"): "#ffd166",
}
SHADES = {"fixed": "#4a4a4a", "interplay": "#6b5410"}
AXIS_LABELS = {"frequency": "Register / brightness", "intensity": "Loudness", "density": "Density"}
MEMORY = 32
GIF_STEP = 3
GIF_FPS = 10
VIDEO_FPS = 10
LIMIT = 2.6
SMOOTH = 32
TRAIL_ALPHA = 0.35
GIF_PALETTE = (
    "split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];"
    "[b][p]paletteuse=dither=none:diff_mode=rectangle"
)


def use_dark() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": FOREGROUND,
            "axes.labelcolor": FOREGROUND,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
        }
    )


def dark_3d(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1))
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID


def build(table: pd.DataFrame, realtime: bool) -> FuncAnimation:
    pos = {n: np.clip(positions(table, n), -LIMIT, LIMIT) for n in INSTRUMENTS}
    times = table.time.to_numpy()
    frames = (
        np.arange(times[0], times[-1], 1 / VIDEO_FPS)
        if realtime
        else np.arange(0, len(table), GIF_STEP)
    )
    index = np.searchsorted(times, frames, side="right") - 1 if realtime else frames

    use_dark()
    fig = plt.figure(figsize=(10, 8.4))
    ax = fig.add_axes([0.02, 0.24, 0.96, 0.7], projection="3d")
    strip = fig.add_axes([0.09, 0.05, 0.85, 0.13])
    for section, (start, end) in SECTIONS.items():
        strip.axvspan(start, end, color=SHADES[section], alpha=0.7, lw=0)
    for x, y in PAIRS:
        strip.plot(
            times,
            table[f"{x}_{y}_twist"].rolling(SMOOTH, center=True, min_periods=8).mean(),
            color=PAIR_COLORS[(x, y)],
            lw=1,
            label=f"{x.capitalize()}–{y.capitalize()}",
        )
    cursor = strip.axvline(0, color=FOREGROUND, lw=2)
    strip.set_xlim(0, times.max())
    strip.set_ylabel("Twist (°/bar)", fontsize=8)
    strip.set_xlabel("Time (s)", fontsize=8)
    strip.tick_params(labelsize=7)
    strip.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")

    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-LIMIT, LIMIT)
    for setter, axis in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), AXES, strict=True):
        setter(AXIS_LABELS[axis], fontsize=8)
    ax.tick_params(labelsize=6)
    dark_3d(ax)
    surfaces = {}
    for pair in PAIRS:
        surfaces[pair] = Line3DCollection([np.zeros((2, 3))], lw=0.9)
        ax.add_collection3d(surfaces[pair], autolim=False)
    trails = {
        n: ax.plot([], [], [], color=COLORS[n], lw=1.2, alpha=TRAIL_ALPHA)[0] for n in INSTRUMENTS
    }
    dots = {
        n: ax.scatter(
            [0],
            [0],
            [0],
            s=120,
            color=COLORS[n],
            edgecolor="white",
            linewidths=0.6,
            depthshade=False,
        )
        for n in INSTRUMENTS
    }
    title = fig.suptitle("", fontsize=11)
    fig.text(
        0.02,
        0.205,
        "Lines join two players at each beat over the last 8 bars (older lines fade). "
        "Moving as one → flat band; moving apart or against → twisted surface.",
        fontsize=7,
        color=MUTED,
    )
    legend = [
        plt.Line2D([], [], color=COLORS[n], marker="o", lw=0, label=n.capitalize())
        for n in INSTRUMENTS
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, frameon=False)
    fade = np.linspace(0.05, 0.9, MEMORY)

    def draw(f: int):
        k = index[f]
        lo = max(0, k - MEMORY + 1)
        for x, y in PAIRS:
            segs = np.stack([pos[x][lo : k + 1], pos[y][lo : k + 1]], axis=1)
            rgba = np.tile(to_rgba(PAIR_COLORS[(x, y)]), (len(segs), 1))
            rgba[:, 3] = fade[-len(segs) :]
            surfaces[(x, y)].set_segments(segs)
            surfaces[(x, y)].set_color(rgba)
        for name in INSTRUMENTS:
            p = pos[name]
            trails[name].set_data(p[lo : k + 1, 0], p[lo : k + 1, 1])
            trails[name].set_3d_properties(p[lo : k + 1, 2])
            dots[name]._offsets3d = ([p[k, 0]], [p[k, 1]], [p[k, 2]])
        ax.view_init(elev=22, azim=-60 + 0.08 * k)
        t = frames[f] if realtime else times[k]
        cursor.set_xdata([t, t])
        section = TITLES.get(table.section.iat[k])
        title.set_text(
            "  ·  ".join(
                filter(
                    None,
                    [
                        f"{int(t // 60)}:{t % 60:04.1f}",
                        section,
                        "ruled surfaces between the players",
                    ],
                )
            )
        )
        return []

    return FuncAnimation(fig, draw, frames=len(frames))


def run(realtime: bool = False) -> None:
    table = pd.read_csv(DATA_DIR / "ruled.csv", index_col="beat")
    anim = build(table, realtime)
    if realtime:
        out = WORK_DIR / "ruled_realtime.mp4"
        silent = out.with_name("ruled_silent.mp4")
        anim.save(silent, writer=FFMpegWriter(fps=VIDEO_FPS), dpi=100)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                silent,
                "-i",
                AUDIO,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                out,
            ],
            check=True,
        )
        silent.unlink()
    else:
        FIG_DIR.mkdir(exist_ok=True)
        out = FIG_DIR / "ruled.gif"
        anim.save(
            out,
            writer=FFMpegWriter(fps=GIF_FPS, extra_args=["-filter_complex", GIF_PALETTE]),
            dpi=60,
        )
    print(out)
