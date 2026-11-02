import inspect
from nodmod import MODSong

song = MODSong()
song.load(input("Song filename: "))

print("=== EFFECT ITERATOR ===")
print(inspect.signature(song.iter_effects))

try:
    print(inspect.getsource(song.iter_effects))
except (OSError, TypeError):
    print("Source unavailable")

print("\n=== EFFECTS IN PLAYBACK ===")
for row_info in list(song.iter_playback_rows())[:100]:
    for ch in range(song.n_channels):
        note = song.get_note(
            row_info.sequence_idx,
            row_info.row,
            ch,
            resolved=True
        )
        if note.effect:
            print(
                f"sequence={row_info.sequence_idx}, "
                f"pattern={row_info.pattern_idx}, "
                f"row={row_info.row}, channel={ch}, "
                f"effect={note.effect}"
            )