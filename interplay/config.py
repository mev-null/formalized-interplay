"""Paths, parts of the track, compared sections and shared parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "audio" / "autumn_leaves.m4a"
WORK_DIR = ROOT / "work"
SEPARATED_DIR = ROOT / "separated"
STEM_DIR = SEPARATED_DIR / "merged"
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

SR = 22050
HOP = 256
TEMPO_PRIOR = 200.0
BEATS_PER_BAR = 4

INSTRUMENTS = ("piano", "bass", "drums")

# Parts for beat estimation: start, end (None = end of track), and the instruments whose onsets
# keep the pulse there. Boundaries are the author's listening (0:08, 0:45, 0:56, 2:00) moved to the
# first onset of the new part: bass at 8.15 (tutti with piano 8.18, drums 8.21), the bass solo's
# first note at 45.37 (piano and drums stop at 44.94), piano re-entry at 56.77, piano at 120.08
# (bass 120.11). Onsets are on an 11.6 ms grid; piano onsets lag by about 40 ms. In the interplay
# the bass plays off the beat, so the drums (from 1:08) are the reference.
PARTS = {
    "intro": (0.0, 8.15, ("piano", "bass", "drums")),
    "theme": (8.15, 45.37, ("bass", "drums")),
    "bass_solo": (45.37, 56.77, ("bass",)),
    "interplay": (56.77, 120.08, ("drums",)),
    "rest": (120.08, None, ("bass", "drums")),
}
# Sections compared in the report (§2.3), bounded by the parts above: fixed roles = intro and
# theme, interplay = from the piano's re-entry to 2:00. The bass solo belongs to neither.
SECTIONS = {
    "fixed": (0.0, PARTS["theme"][1]),
    "interplay": (PARTS["interplay"][0], PARTS["interplay"][1]),
}
TITLES = {"fixed": "Fixed roles", "interplay": "Interplay"}

# Beats this close to a part boundary (seconds) are transitions, left out of section comparisons.
TRANSITION = 1.2


def section_of(t: float, transition: bool = False) -> str:
    if transition:
        return "transition"
    for name, (start, end) in SECTIONS.items():
        if start <= t < end:
            return name
    return "other"
