"""Run the pipeline: `uv run python -m interplay [step ...]` (default: all report steps)."""

import argparse

from . import animation, figures, grains, ruled, screen_space, separate, transcribe

STEPS = {
    "separate": separate.run,
    "transcribe": transcribe.run,
    "grains": grains.run,
    "screen-space": screen_space.run,
    "ruled": ruled.run,
    "figures": figures.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m interplay", description=__doc__)
    parser.add_argument(
        "steps",
        nargs="*",
        choices=[*STEPS, "animation", "all"],
        default=["all"],
        help="steps to run, in pipeline order (default: all)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="animation: real-time MP4 with the local audio (private, never publish)",
    )
    args = parser.parse_args()
    chosen = list(STEPS) if "all" in args.steps else [s for s in STEPS if s in args.steps]
    for step in chosen:
        print(f"== {step}")
        STEPS[step]()
    if "animation" in args.steps:
        print("== animation")
        animation.run(args.realtime)


if __name__ == "__main__":
    main()
