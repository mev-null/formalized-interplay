"""Analysis behind the report's figures, run in steps by `python -m interplay`.

1. `separate`: piano, bass and drums from the recording (Demucs)
2. `transcribe`: piano notes
3. `grains`: sound events of each player, and the beat grid (`beats`)
4. `screen_space`: each player per beat as a point (frequency, loudness, density)
5. `ruled`: rulings between pairs of players and their twist
6. `figures`: Fig. 1 to Fig. 3

`animation` draws the ruled surfaces over the whole track (not in the report). Paths, parts,
sections and parameters are in `config`.
"""
