from nodmod import MODSong
from mido import MidiFile, MidiTrack, Message, MetaMessage
import os

song = MODSong()
filename = input("Song filename: ")
print("")
song.load(filename)

os.makedirs("sample_out", exist_ok=True)

used = song.get_used_samples(scope="reachable", order="sorted")
for sample_idx in used:
    try:
        song.save_sample(sample_idx, f"sample_out/sample_{sample_idx}.wav")
    except Exception:
        pass

playback = list(song.iter_playback_rows(max_steps=250_000))
if not playback:
    raise ValueError("No playback rows")

n_channels = song.n_channels
active = [None] * n_channels
events_by_sample = {}

def midi_note(period_str):
    if not period_str:
        return None
    try:
        idx = song.note_to_index(period_str)
        return idx + 24
    except Exception:
        return None

def close_note(ch, end_sec):
    info = active[ch]
    if info is None:
        return
    sample_idx, note_num, start_sec = info
    if end_sec <= start_sec:
        end_sec = start_sec + 0.01
    events_by_sample.setdefault(sample_idx, []).append((start_sec, end_sec, note_num))
    active[ch] = None

for row_info in playback:
    seq_idx = row_info.sequence_idx
    row = row_info.row
    start_sec = row_info.start_sec
    end_sec = row_info.end_sec

    for ch in range(n_channels):
        note = song.get_note(seq_idx, row, ch, resolved=True)
        period = note.period if note.period else ""
        sample_idx = note.instrument_idx
        effect = (note.effect or "").upper()

        is_note_cut = effect.startswith("EC") or effect.startswith("C0")
        has_new_note = bool(period) and sample_idx > 0

        if has_new_note or is_note_cut:
            close_note(ch, start_sec)

        if has_new_note:
            midi_n = midi_note(period)
            if midi_n is not None:
                active[ch] = (sample_idx, midi_n, start_sec)

song_end = playback[-1].end_sec if playback else 0.0
for ch in range(n_channels):
    close_note(ch, song_end)

mid = MidiFile(ticks_per_beat=480)
tempo = 500000

for sample_idx in sorted(events_by_sample.keys()):
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("track_name", name=f"sample {sample_idx}", time=0))
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(Message("program_change", program=0, time=0))

    events = []
    for start_sec, end_sec, note_num in events_by_sample[sample_idx]:
        events.append((start_sec, 1, note_num))
        events.append((end_sec, 0, note_num))
    events.sort(key=lambda x: (x[0], x[1]))

    last_tick = 0
    for t_sec, is_on, note_num in events:
        tick = int(round(t_sec * 1_000_000 / tempo * mid.ticks_per_beat))
        delta = max(0, tick - last_tick)
        if is_on:
            track.append(Message("note_on", note=note_num, velocity=100, time=delta))
        else:
            track.append(Message("note_off", note=note_num, velocity=0, time=delta))
        last_tick = tick

out_name = os.path.splitext(os.path.basename(filename))[0] + ".mid"
mid.save(out_name)
print(f"Wrote {out_name}")
