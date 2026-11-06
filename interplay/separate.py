"""Step 1: split the recording into piano, bass and drums with Demucs.

The bass is panned right in this recording. Separating the stereo mix puts almost all of it into
Demucs' "other" stem, so each channel is separated on its own and the stems are rebuilt as stereo.
Demucs shifts the input by a random offset before separating, so the seed is fixed for
repeatable stems.
"""

import random
import subprocess

import demucs.separate
import numpy as np
import soundfile as sf

from .config import AUDIO, SEPARATED_DIR, STEM_DIR, WORK_DIR

MODEL = "htdemucs_ft"
SEED = 0
STEMS = {"piano": "other", "bass": "bass", "drums": "drums"}
CHANNELS = {"L": "c0", "R": "c1"}


def split_channels() -> list:
    WORK_DIR.mkdir(exist_ok=True)
    paths = []
    for name, channel in CHANNELS.items():
        path = WORK_DIR / f"channel_{name}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                AUDIO,
                "-af",
                f"pan=stereo|c0={channel}|c1={channel}",
                path,
            ],
            check=True,
        )
        paths.append(path)
    return paths


def merge() -> None:
    STEM_DIR.mkdir(parents=True, exist_ok=True)
    source = SEPARATED_DIR / MODEL
    for name, stem in STEMS.items():
        left, sr = sf.read(source / "channel_L" / f"{stem}.wav")
        right, _ = sf.read(source / "channel_R" / f"{stem}.wav")
        sf.write(
            STEM_DIR / f"{name}.wav",
            np.stack([left.mean(axis=1), right.mean(axis=1)], axis=1),
            sr,
            subtype="FLOAT",
        )


def run() -> None:
    paths = split_channels()
    random.seed(SEED)
    demucs.separate.main(["-n", MODEL, "-d", "cpu", "-o", str(SEPARATED_DIR), *map(str, paths)])
    merge()
