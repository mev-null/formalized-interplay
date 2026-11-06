"""Step 2: transcribe the piano stem into notes (onset, pitch, velocity).

Model: High-resolution Piano Transcription (Kong et al. 2021). The weights (CC BY 4.0) are
downloaded from Zenodo and checked against their SHA-256; they are not redistributed here.
"""

import hashlib
import urllib.request

import librosa
import pandas as pd

from .config import DATA_DIR, MODEL_DIR, STEM_DIR

WEIGHTS_URL = (
    "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
WEIGHTS = MODEL_DIR / "piano_transcription_note_F1=0.9677_pedal_F1=0.9186.pth"
WEIGHTS_SHA256 = "c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141"
NOTES = DATA_DIR / "piano_notes.csv"


def download_weights() -> None:
    if not WEIGHTS.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError(f"checksum mismatch for {WEIGHTS}")


def run() -> None:
    from piano_transcription_inference import PianoTranscription, sample_rate

    download_weights()
    DATA_DIR.mkdir(exist_ok=True)
    audio, _ = librosa.load(STEM_DIR / "piano.wav", sr=sample_rate, mono=True)
    model = PianoTranscription(device="cpu", checkpoint_path=str(WEIGHTS))
    result = model.transcribe(audio, str(NOTES.with_suffix(".mid")))
    notes = pd.DataFrame(result["est_note_events"]).rename(
        columns={"onset_time": "onset", "offset_time": "offset", "midi_note": "pitch"}
    )
    notes[["onset", "offset", "pitch", "velocity"]].sort_values("onset").to_csv(NOTES, index=False)
