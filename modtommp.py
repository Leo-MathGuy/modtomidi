#!/usr/bin/env python3
"""Convert tracker modules (.mod, .xm, .s3m) into LMMS projects (.mmp).

The resulting project contains one AudioFileProcessor track per used sample.
Every sample is written to ``<folder>/samples/sampleN.wav`` and the project
file ``<folder>/song.mmp`` references those files with relative paths.

Notes, volume, panning and pitch effects are imported from the module.  Pitch
changes that happen inside a note (portamento, vibrato, arpeggio, ...) are
written as per-note "Note detuning" automation clips, which is how LMMS can
pitch-shift a single note.
"""

from __future__ import annotations

import math
import os
import sys
import wave
from array import array
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from nodmod import Song, load_song
from nodmod.modsong import MODSong

# LMMS uses 192 ticks per 4/4 bar (48 ticks per quarter note).
TICKS_PER_BAR = 192
# Sounding pitch of key 72 (C-5) is the reference pitch of the imported samples.
ROOT_KEY = 72
# LMMS tick duration is 1.25 / BPM seconds.
SECONDS_PER_TICK_AT_BPM = 1.25

VOLUME_MAX = 64
PAN_CENTER = 128


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def fmt_number(value) -> str:
    """Format a float compactly for the XML attributes LMMS writes."""
    if isinstance(value, int):
        return str(value)
    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def note_pitch_and_octave(period: str):
    """Split a tracker note such as ``C#5`` into (semitone, octave)."""
    text = period.strip().upper()
    if len(text) < 3 or text[1] not in ("-", "#"):
        return None
    pitch = text[:2]
    if pitch not in Song.PERIOD_SEQ:
        return None
    try:
        octave = int(text[2:])
    except ValueError:
        return None
    return Song.PERIOD_SEQ.index(pitch), octave


def tracker_note_index(period: str, fmt: str):
    """Return a tracker-style note index (C-5 == 48) for MOD/XM/S3M notes.

    nodmod represents S3M notes one octave lower than the tracker convention
    (S3M note byte 0x40 is C-5 in ST3 but "C-4" in nodmod), so S3M gets a
    one-octave correction here.
    """
    parsed = note_pitch_and_octave(period)
    if parsed is None:
        return None
    pitch, octave = parsed
    if fmt == "s3m":
        return octave * 12 + pitch
    return (octave - 1) * 12 + pitch


def lmms_key_for_period(period: str, fmt: str):
    index = tracker_note_index(period, fmt)
    if index is None:
        return None
    # C-5 (tracker index 48) becomes LMMS key 72.
    return clamp(24 + index, 0, 127)


def volume_to_lmms(volume: float) -> int:
    return int(round(clamp(volume, 0, VOLUME_MAX) * 100.0 / VOLUME_MAX))


def pan_to_lmms(pan255: float) -> int:
    return int(round(clamp((pan255 - PAN_CENTER) / 127.0 * 100.0, -100, 100)))


def s3m_pan_to_255(pan: float) -> float:
    """S3M panning is 0..128, while MOD/XM use 0..255."""
    return clamp(pan, 0, 128) * 255.0 / 128.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SampleAsset:
    key: int                      # format-specific sample identifier
    name: str
    sample: object                # nodmod Sample / XMSample / S3MSample
    fmt: str
    volume: float = 64.0          # tracker default volume 0..64
    pan: int = PAN_CENTER         # 0..255 (used as the initial pan for XM)
    loop_start: int = 0
    loop_length: int = 0
    length: int = 0
    filename: str = ""
    track_name: str = ""


@dataclass
class Segment:
    asset: SampleAsset
    start_tick: int
    key: int
    volume: float
    pan: int                      # LMMS -100..100
    end_tick: int | None = None
    detune: list[tuple[int, float]] = field(default_factory=list)

    def add_detune(self, tick: int, value: float):
        """Append a detuning automation point, avoiding duplicate values."""
        relative = max(0, int(tick) - self.start_tick)
        value = clamp(float(value), -60.0, 60.0)
        if self.detune:
            last_tick, last_value = self.detune[-1]
            if relative < last_tick:
                return
            if relative == last_tick:
                self.detune[-1] = (relative, value)
                return
            if abs(last_value - value) < 0.005:
                return
        self.detune.append((relative, value))

    def finish(self, end_tick: int):
        if end_tick <= self.start_tick:
            end_tick = self.start_tick + 1
        self.end_tick = max(self.start_tick + 1, int(end_tick))

    @property
    def length(self) -> int:
        if self.end_tick is None:
            return 1
        return max(1, self.end_tick - self.start_tick)


@dataclass
class ChannelState:
    index: int
    asset: SampleAsset | None = None
    instrument: int = 0           # XM instrument / S3M sample memory
    note_index: int | None = None
    key: int | None = None
    period: float = 0.0           # current period value
    base_period: float = 0.0
    target_period: float | None = None
    pitch_adjust: float = 0.0     # semitone offset from finetune effects
    volume: float = VOLUME_MAX
    global_volume: float = VOLUME_MAX
    pan: float = PAN_CENTER
    pan_explicit: bool = False
    segment: Segment | None = None
    vibrato_speed: int = 0
    vibrato_depth: int = 0
    vibrato_phase: float = 0.0
    vibrato_waveform: int = 0
    tremolo_speed: int = 0
    tremolo_depth: int = 0
    tremolo_phase: float = 0.0
    tremolo_waveform: int = 0
    arp_x: int = 0
    arp_y: int = 0
    memory: dict = field(default_factory=dict)
    pitch_automation: bool = False


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class MMPConverter:
    def __init__(self, song, module_path: str, output_dir: str):
        self.song = song
        self.fmt = song.file_extension
        self.module_path = module_path
        self.output_dir = output_dir
        self.samples_dir = os.path.join(output_dir, "samples")
        self.assets: dict[int, SampleAsset] = {}
        self.assets_by_key: list[SampleAsset] = []
        self.segments_by_key: dict[int, list[Segment]] = {}
        self.channels: list[ChannelState] = []
        self.base_bpm = 125.0
        self.linear_xm = bool(getattr(song, "uses_linear_frequency", False))
        self.global_volume = 64.0
        self.used_track_names: set[str] = set()
        # XM sample flattening maps.
        self.xm_flat_map: dict[int, tuple[int, int, object]] = {}
        self.xm_inst_flat: dict[int, list[int]] = {}

    # -- public entry point -------------------------------------------------

    def run(self):
        os.makedirs(self.samples_dir, exist_ok=True)
        self._collect_assets()
        rows = list(self.song.iter_playback_rows(max_steps=250_000))
        if not rows:
            raise ValueError("The module contains no playable rows.")
        self.base_bpm = float(rows[0].tempo or 125)
        self._init_channels()
        self._process_rows(rows)
        self._finish_notes(rows)
        self._export_samples()
        self._write_project()

    # -- sample inventory ---------------------------------------------------

    def _collect_assets(self):
        """Register every non-empty sample slot.

        Assets that end up with no imported notes are dropped again when the
        samples are exported, so the output folder still only contains samples
        that the song actually uses.
        """
        if self.fmt == "xm":
            flat_idx = 1
            for inst_idx, inst in enumerate(self.song.instruments, start=1):
                flats = []
                for local_idx, sample in enumerate(inst.samples, start=1):
                    self.xm_flat_map[flat_idx] = (inst_idx, local_idx, sample)
                    flats.append(flat_idx)
                    self._add_asset(
                        flat_idx,
                        sample,
                        sample.name,
                        sample.panning,
                        sample.volume,
                    )
                    flat_idx += 1
                self.xm_inst_flat[inst_idx] = flats
        else:
            for sample_idx, sample in enumerate(self.song.samples, start=1):
                self._add_asset(
                    sample_idx,
                    sample,
                    getattr(sample, "name", ""),
                    PAN_CENTER,
                    getattr(sample, "volume", 64),
                )

    def _add_asset(self, key, sample, name, pan, volume):
        length = len(getattr(sample, "waveform", []))
        if length == 0:
            return
        loop_start = int(getattr(sample, "repeat_point", 0) or 0)
        loop_length = int(getattr(sample, "repeat_len", 0) or 0)
        loop_active = loop_length > 1 and 0 <= loop_start < length
        if loop_active:
            loop_active = loop_start + loop_length <= length
        if loop_active and self.fmt == "mod":
            # ProTracker-style modules commonly use a one-word (two sample)
            # loop to mean "no loop".
            loop_active = loop_length > 2
        elif loop_active and self.fmt == "xm":
            loop_active = int(getattr(sample, "loop_type", 0) or 0) != 0
        elif loop_active and self.fmt == "s3m":
            loop_active = bool(int(getattr(sample, "flags", 0) or 0) & 0x01)
        if not loop_active:
            loop_start = 0
            loop_length = 0
        base_name = (name or f"sample {key}").strip()
        asset = SampleAsset(
            key=key,
            name=base_name,
            sample=sample,
            fmt=self.fmt,
            volume=float(clamp(volume, 0, VOLUME_MAX)),
            pan=int(clamp(pan, 0, 255)),
            loop_start=loop_start,
            loop_length=loop_length,
            length=length,
        )
        self.assets[key] = asset

    # -- playback state -----------------------------------------------------

    def _init_channels(self):
        n_channels = int(self.song.n_channels)
        self.channels = [ChannelState(index=i) for i in range(n_channels)]
        global_volume = float(getattr(self.song, "global_volume", 64) or 64)
        self.global_volume = clamp(global_volume, 0, 64)

        for channel in self.channels:
            channel.global_volume = self.global_volume
            if self.fmt == "s3m":
                channel.pan = self._s3m_initial_pan(channel.index)

    def _s3m_initial_pan(self, compact_channel: int) -> float:
        try:
            raw = self.song.compact_to_raw_channel[compact_channel]
            setting = self.song.channel_settings[raw] & 0x7F
        except (AttributeError, IndexError):
            return PAN_CENTER
        # Prefer the explicit S3M default-panning table when the file has one.
        panning = getattr(self.song, "default_panning", None)
        if (
            getattr(self.song, "default_pan_flag", 0) == 252
            and isinstance(panning, list)
            and len(panning) == 32
        ):
            return clamp(float(panning[raw] & 0x0F) * 255.0 / 15.0, 0, 255)
        # Otherwise raw channel slots 0-7 are left and 8-15 are right.
        if 0 <= setting <= 7:
            return 0.0
        if 8 <= setting <= 15:
            return 255.0
        return PAN_CENTER

    def _new_channel_volume(self, channel: ChannelState, asset: SampleAsset) -> float:
        return asset.volume

    def _resolve_asset(self, channel: ChannelState, note, period: str):
        """Resolve which sample a (possibly implicit) note refers to."""
        if self.fmt == "xm":
            if note.instrument_idx > 0:
                channel.instrument = int(note.instrument_idx)
            if not period:
                return channel.asset
            inst_idx = channel.instrument
            if inst_idx <= 0 or inst_idx > len(self.song.instruments):
                return channel.asset
            flat = self._xm_flat_for_note(inst_idx, period)
            if flat is not None:
                asset = self.assets.get(flat)
                if asset is not None:
                    channel.asset = asset
                    return asset
            return channel.asset

        if self.fmt == "s3m":
            if note.instrument_idx > 0:
                channel.instrument = int(note.instrument_idx)
            if note.instrument_idx > 0:
                channel.asset = self.assets.get(channel.instrument)
            if not period:
                return channel.asset
            return channel.asset

        # MOD: get_note(..., resolved=True) already applied sample memory.
        if note.instrument_idx > 0:
            channel.instrument = int(note.instrument_idx)
            channel.asset = self.assets.get(channel.instrument)
        return channel.asset

    def _xm_flat_for_note(self, inst_idx: int, period: str):
        flats = self.xm_inst_flat.get(inst_idx, [])
        if not flats:
            return None
        if len(flats) == 1:
            return flats[0]
        inst = self.song.instruments[inst_idx - 1]
        if period and period != "off" and len(inst.sample_map) == 96:
            try:
                note_idx = Song.note_to_index(period)
            except (TypeError, ValueError):
                note_idx = None
            if note_idx is not None and 0 <= note_idx < 96:
                local = inst.sample_map[note_idx]
                if 0 <= local < len(flats):
                    return flats[local]
        return flats[0]

    # -- timing -------------------------------------------------------------

    def _seconds_to_tick(self, seconds: float) -> int:
        return int(round(seconds * self.base_bpm / SECONDS_PER_TICK_AT_BPM))

    def _row_ticks(self, row):
        return self._seconds_to_tick(row.start_sec), self._seconds_to_tick(row.end_sec)

    def _tick_in_row(self, row, row_start_tick, row_end_tick, tick_number):
        speed = max(1, int(row.speed))
        tick_number = clamp(tick_number, 0, speed)
        span = max(1, row_end_tick - row_start_tick)
        return row_start_tick + int(round(span * tick_number / speed))

    # -- note/pitch helpers -------------------------------------------------

    def _period_for_note(self, note_index: int) -> float:
        if self.fmt == "xm" and self.linear_xm:
            # FastTracker linear periods: 64 units per semitone.
            return 7680.0 - 64.0 * note_index
        if self.fmt == "s3m":
            # S3M periods are four times the Amiga period values.
            return 1712.0 * (2.0 ** (-(note_index - 48) / 12.0))
        # MOD / XM Amiga: use the exact ProTracker period table when possible.
        try:
            name = Song.index_to_note(note_index)
            return float(MODSong.INV_PERIOD_TABLE[name])
        except (IndexError, KeyError, ValueError):
            return 428.0 * (2.0 ** (-(note_index - 48) / 12.0))

    def _period_to_detune(self, base_period: float, period: float) -> float:
        if base_period <= 0 or period <= 0:
            return 0.0
        if self.fmt == "xm" and self.linear_xm:
            return (base_period - period) / 64.0
        return 12.0 * math.log2(base_period / period)

    def _vibrato_wave(self, phase: float, waveform: int) -> float:
        phase = phase % 64.0
        radians = phase / 64.0 * 2.0 * math.pi
        if waveform == 1:  # ramp down
            return 1.0 - 2.0 * (phase / 64.0)
        if waveform == 2:  # square
            return 1.0 if phase < 32 else -1.0
        return math.sin(radians)

    # -- segment handling ---------------------------------------------------

    def _finish_segment(self, channel: ChannelState, tick: int):
        segment = channel.segment
        if segment is None:
            return
        if channel.key is not None:
            value = self._current_detune(channel)
            segment.add_detune(tick, value)
        segment.finish(tick)
        channel.segment = None

    def _effective_volume(self, channel: ChannelState) -> float:
        return channel.volume * channel.global_volume / 64.0

    def _start_segment(self, channel: ChannelState, tick: int, key: int):
        asset = channel.asset
        if asset is None:
            return
        segment = Segment(
            asset=asset,
            start_tick=int(tick),
            key=int(key),
            volume=self._effective_volume(channel),
            pan=pan_to_lmms(channel.pan),
        )
        if channel.pitch_adjust:
            segment.add_detune(tick, channel.pitch_adjust)
        self.segments_by_key.setdefault(asset.key, []).append(segment)
        channel.segment = segment

    def _restart_segment(self, channel: ChannelState, tick: int):
        """Restart the active note to reflect an explicit volume/pan change."""
        if channel.segment is None or channel.key is None:
            return
        if channel.segment.start_tick >= tick:
            return
        self._finish_segment(channel, tick)
        self._start_segment(channel, tick, channel.key)
        if channel.segment is not None:
            channel.segment.add_detune(tick, self._current_detune(channel))

    def _current_detune(self, channel: ChannelState) -> float:
        if channel.base_period <= 0 or channel.period <= 0:
            return channel.pitch_adjust
        return self._period_to_detune(channel.base_period, channel.period) + channel.pitch_adjust

    def _record_pitch(self, channel: ChannelState, tick: int, modulation: float = 0.0):
        segment = channel.segment
        if segment is None:
            return
        value = self._current_detune(channel) + modulation
        if abs(value) < 0.005 and not channel.pitch_automation and not segment.detune:
            return
        segment.add_detune(tick, value)
        if abs(value) >= 0.005:
            channel.pitch_automation = True

    # -- main row processing ------------------------------------------------

    def _process_rows(self, rows):
        for row in rows:
            self._process_row(row)

    def _finish_notes(self, rows):
        end_tick = self._seconds_to_tick(rows[-1].end_sec)
        for channel in self.channels:
            self._finish_segment(channel, end_tick)

    def _process_row(self, row):
        row_start, row_end = self._row_ticks(row)
        speed = max(1, int(row.speed))
        for channel in self.channels:
            try:
                if self.fmt == "mod":
                    note = self.song.get_note(
                        row.sequence_idx, row.row, channel.index, resolved=True
                    )
                else:
                    note = self.song.get_note(row.sequence_idx, row.row, channel.index)
            except (IndexError, TypeError, ValueError):
                continue
            self._process_cell(channel, row, note, row_start, row_end, speed)

    def _process_cell(self, channel, row, note, row_start, row_end, speed):
        period = (getattr(note, "period", "") or "").strip()
        effect = (getattr(note, "effect", "") or "").strip().upper()
        command, arg, x, y = self._parse_effect(effect)

        vol_cmd = getattr(note, "vol_cmd", None)
        vol_val = getattr(note, "vol_val", -1)
        s3m_volume = getattr(note, "volume", -1)

        # --- figure out explicit set commands, needed before a note starts ---
        set_volume = None
        if vol_cmd == "v" and vol_val >= 0:
            set_volume = clamp(float(vol_val), 0, VOLUME_MAX)
        elif self.fmt == "s3m" and isinstance(s3m_volume, int) and s3m_volume >= 0:
            set_volume = clamp(float(s3m_volume), 0, VOLUME_MAX)
        if self.fmt in ("mod", "xm") and command == "C" and arg is not None:
            set_volume = float(arg)

        set_pan = None
        if vol_cmd == "p" and vol_val >= 0:
            set_pan = clamp(float(vol_val) * 255.0 / 60.0, 0, 255)
        if command == "8" and arg is not None:
            set_pan = float(arg)
        elif self.fmt == "s3m" and command == "X" and arg is not None:
            set_pan = s3m_pan_to_255(arg)

        # --- classify note --------------------------------------------------
        is_off = period == "off" or (self.fmt == "xm" and command == "K")
        has_note = bool(period) and period != "off"
        tone_porta = self._is_tone_portamento(command, vol_cmd)

        delay_ticks = 0
        cut_tick = None
        if command == "E" and x is not None:
            if x == 0xD:  # note delay
                delay_ticks = y
            elif x == 0xC:  # note cut
                cut_tick = self._tick_in_row(row, row_start, row_end, y)
        elif self.fmt == "s3m" and command == "S" and x is not None:
            if x == 0xD:
                delay_ticks = y
            elif x == 0xC:
                cut_tick = self._tick_in_row(row, row_start, row_end, y)

        trigger_tick = self._tick_in_row(row, row_start, row_end, delay_ticks)
        # A note cut on tick 0 means the note never sounds at all.
        cut_before_start = cut_tick is not None and cut_tick <= trigger_tick

        asset = None
        note_index = None
        key = None
        if has_note:
            note_index = tracker_note_index(period, self.fmt)
            key = lmms_key_for_period(period, self.fmt)
            asset = self._resolve_asset(channel, note, period)
        else:
            # Instrument/sample memory can change even without a note.
            self._resolve_asset(channel, note, "")

        tone_porta_active = (
            tone_porta and channel.segment is not None and channel.note_index is not None
        )

        # --- start / finish notes ------------------------------------------
        if is_off:
            self._finish_segment(channel, row_start)
        elif has_note and key is not None and asset is not None:
            if tone_porta_active:
                # Tone portamento does not retrigger the sample; it glides to
                # the new note instead.
                channel.target_period = self._period_for_note(note_index)
                if set_volume is not None:
                    channel.segment.volume = set_volume * channel.global_volume / 64.0
                if set_pan is not None:
                    channel.segment.pan = pan_to_lmms(set_pan)
            elif cut_before_start:
                self._finish_segment(channel, trigger_tick)
                channel.asset = asset
                channel.note_index = note_index
                channel.key = key
            else:
                self._finish_segment(channel, trigger_tick)
                channel.asset = asset
                channel.note_index = note_index
                channel.key = key
                channel.base_period = self._period_for_note(note_index)
                channel.period = channel.base_period
                channel.target_period = None
                channel.pitch_adjust = 0.0
                channel.pitch_automation = False
                channel.vibrato_phase = 0.0
                channel.tremolo_phase = 0.0
                channel.volume = (
                    set_volume
                    if set_volume is not None
                    else self._new_channel_volume(channel, asset)
                )
                if set_pan is not None:
                    channel.pan = set_pan
                elif not channel.pan_explicit and self.fmt == "xm":
                    channel.pan = asset.pan
                self._start_segment(channel, trigger_tick, key)

        # Apply explicit volume / pan changes to an already sounding note.
        if set_volume is not None:
            channel.volume = set_volume
            if channel.segment is not None:
                effective = set_volume * channel.global_volume / 64.0
                if tone_porta_active:
                    channel.segment.volume = effective
                elif channel.segment.start_tick == trigger_tick:
                    channel.segment.volume = effective
                elif channel.segment.start_tick < trigger_tick:
                    self._restart_segment(channel, trigger_tick)
                    if channel.segment is not None:
                        channel.segment.volume = effective
        if set_pan is not None:
            channel.pan = set_pan
            channel.pan_explicit = True
            if channel.segment is not None:
                if tone_porta_active:
                    channel.segment.pan = pan_to_lmms(set_pan)
                elif channel.segment.start_tick == trigger_tick:
                    channel.segment.pan = pan_to_lmms(set_pan)
                elif channel.segment.start_tick < trigger_tick:
                    self._restart_segment(channel, trigger_tick)
                    if channel.segment is not None:
                        channel.segment.pan = pan_to_lmms(set_pan)

        # --- continuous-effect state for this row --------------------------
        slide_period = 0.0        # period units per tick, positive = down
        porta_speed = None        # glide towards target_period
        vol_slide_up = 0
        vol_slide_down = 0
        arp = None
        vibrato_active = False

        if command is not None:
            if self.fmt in ("mod", "xm"):
                if command == "1" and arg:
                    channel.memory["1"] = arg
                    slide_period -= channel.memory["1"]
                elif command == "2" and arg:
                    channel.memory["2"] = arg
                    slide_period += channel.memory["2"]
                elif command in ("1", "2") and arg == 0:
                    slide_period += (
                        -channel.memory.get("1", 0)
                        if command == "1"
                        else channel.memory.get("2", 0)
                    )

                if command == "3":
                    if arg:
                        channel.memory["3"] = arg
                    porta_speed = channel.memory.get("3", 0)
                elif command == "5":
                    if arg:
                        channel.memory["5"] = arg
                    value = channel.memory.get("5", 0)
                    vol_slide_up, vol_slide_down = value >> 4, value & 0x0F
                    porta_speed = channel.memory.get("3", 0)
                elif command == "6":
                    value = arg
                    vol_slide_up, vol_slide_down = value >> 4, value & 0x0F
                    vibrato_active = True
                elif command == "4":
                    if x:
                        channel.vibrato_speed = x
                    if y:
                        channel.vibrato_depth = y
                    vibrato_active = True
                elif command == "A":
                    if arg:
                        channel.memory["A"] = arg
                    value = channel.memory.get("A", 0)
                    vol_slide_up, vol_slide_down = value >> 4, value & 0x0F
                elif command == "0":
                    arp = (x, y)
                elif command == "G":
                    self.global_volume = clamp(float(arg), 0, 64)
                    for other in self.channels:
                        other.global_volume = self.global_volume
                elif command == "E" and x is not None:
                    self._apply_extended(channel, x, y)
                    if x == 0x1:      # fine portamento up
                        channel.period = max(1.0, channel.period - y)
                    elif x == 0x2:    # fine portamento down
                        channel.period += y
                    elif x == 0x4:
                        channel.vibrato_waveform = y
                    elif x == 0x7:
                        channel.tremolo_waveform = y
                    elif x == 0x5:
                        # E5x sets the finetune; repeated rows with the same
                        # value must not keep stacking pitch offsets.
                        previous = channel.memory.get("E5")
                        if previous is None:
                            channel.pitch_adjust += y / 8.0
                        else:
                            channel.pitch_adjust += (y - previous) / 8.0
                        channel.memory["E5"] = y
                elif command == "7":
                    if x:
                        channel.tremolo_speed = x
                    if y:
                        channel.tremolo_depth = y
            else:  # S3M
                if command == "D":
                    if arg:
                        channel.memory["D"] = arg
                    value = channel.memory.get("D", 0)
                    vol_slide_up, vol_slide_down = value >> 4, value & 0x0F
                elif command == "E":
                    if arg:
                        channel.memory["E"] = arg
                    slide_period += channel.memory["E"]
                elif command == "F":
                    if arg:
                        channel.memory["F"] = arg
                    slide_period -= channel.memory["F"]
                elif command == "G":
                    if arg:
                        channel.memory["G"] = arg
                    porta_speed = channel.memory.get("G", 0)
                elif command == "H":
                    if x:
                        channel.vibrato_speed = x
                    if y:
                        channel.vibrato_depth = y
                    vibrato_active = True
                elif command == "J":
                    arp = (x, y)
                elif command == "K":
                    vol_slide_up, vol_slide_down = x, y
                    vibrato_active = True
                elif command == "L":
                    if arg:
                        channel.memory["L"] = arg
                    value = channel.memory.get("L", 0)
                    vol_slide_up, vol_slide_down = value >> 4, value & 0x0F
                    porta_speed = channel.memory.get("G", 0)
                elif command == "U":
                    if x:
                        channel.vibrato_speed = x
                    if y:
                        channel.vibrato_depth = y
                    vibrato_active = True
                    channel.vibrato_waveform = 0
                elif command == "V":
                    self.global_volume = clamp(float(arg), 0, 64)
                    for other in self.channels:
                        other.global_volume = self.global_volume
                elif command == "S":
                    if x == 0x2:
                        previous = channel.memory.get("S2")
                        if previous is None:
                            channel.pitch_adjust += y / 8.0
                        else:
                            channel.pitch_adjust += (y - previous) / 8.0
                        channel.memory["S2"] = y
                    elif x == 0x3:
                        channel.vibrato_waveform = y & 0x03
                    elif x == 0x4:
                        channel.tremolo_waveform = y & 0x03
                    elif x == 0x8:
                        channel.pan = s3m_pan_to_255(y * 8)
                        channel.pan_explicit = True

        # Volume-column slides / vibrato parameters.
        if vol_cmd == "d":
            vol_slide_down = max(vol_slide_down, int(vol_val))
        elif vol_cmd == "c":
            vol_slide_up = max(vol_slide_up, int(vol_val))
        elif vol_cmd == "b":
            channel.volume = clamp(channel.volume - vol_val, 0, 64)
        elif vol_cmd == "a":
            channel.volume = clamp(channel.volume + vol_val, 0, 64)
        elif vol_cmd == "u":
            channel.vibrato_speed = int(vol_val)
        elif vol_cmd == "h":
            channel.vibrato_depth = int(vol_val)
        elif vol_cmd == "g":
            if command != "3" and command != "5":
                porta_speed = channel.memory.get("3", int(vol_val) or 0)
        elif vol_cmd == "l":
            channel.pan = clamp(channel.pan - vol_val * 4, 0, 255)
            channel.pan_explicit = True
        elif vol_cmd == "r":
            channel.pan = clamp(channel.pan + vol_val * 4, 0, 255)
            channel.pan_explicit = True

        # --- per-tick updates ---------------------------------------------
        any_pitch = bool(
            slide_period
            or porta_speed
            or arp is not None
            or vibrato_active
            or channel.pitch_automation
        )

        segment = channel.segment
        if segment is not None:
            # Ensure the note has a value at its own start tick.
            if trigger_tick >= segment.start_tick:
                self._record_pitch(channel, trigger_tick, 0.0)

        for tick_no in range(0, speed):
            tick_pos = self._tick_in_row(row, row_start, row_end, tick_no)

            if tick_no > 0:
                if slide_period:
                    channel.period = max(1.0, channel.period + slide_period)
                if porta_speed and channel.target_period is not None:
                    if channel.period < channel.target_period:
                        channel.period = min(
                            channel.target_period, channel.period + porta_speed
                        )
                    elif channel.period > channel.target_period:
                        channel.period = max(
                            channel.target_period, channel.period - porta_speed
                        )

                if vol_slide_up or vol_slide_down:
                    if vol_slide_up:
                        channel.volume = clamp(channel.volume + vol_slide_up, 0, 64)
                    elif vol_slide_down:
                        channel.volume = clamp(channel.volume - vol_slide_down, 0, 64)

                if vibrato_active:
                    channel.vibrato_phase = (
                        channel.vibrato_phase + channel.vibrato_speed
                    ) % 64.0
                if channel.tremolo_speed:
                    channel.tremolo_phase = (
                        channel.tremolo_phase + channel.tremolo_speed
                    ) % 64.0

            modulation = 0.0
            if arp is not None:
                phase = tick_no % 3
                if phase == 1:
                    modulation += arp[0]
                elif phase == 2:
                    modulation += arp[1]
            if vibrato_active and channel.vibrato_depth:
                modulation += (
                    channel.vibrato_depth
                    / 16.0
                    * self._vibrato_wave(
                        channel.vibrato_phase, channel.vibrato_waveform
                    )
                )

            if tick_no > 0 or any_pitch or modulation:
                self._record_pitch(channel, tick_pos, modulation)

        # --- note cut / delay ---------------------------------------------
        if cut_tick is not None:
            self._finish_segment(channel, min(cut_tick, row_end))

    @staticmethod
    def _is_tone_portamento(command, vol_cmd) -> bool:
        if vol_cmd == "g":
            return True
        if command in ("3", "5"):
            return True
        if command in ("G", "L"):
            return True
        return False

    @staticmethod
    def _parse_effect(effect: str):
        if not effect or len(effect) != 3:
            return None, None, None, None
        command = effect[0]
        try:
            arg = int(effect[1:], 16)
        except ValueError:
            return None, None, None, None
        return command, arg, arg >> 4, arg & 0x0F

    @staticmethod
    def _apply_extended(channel, x, y):
        if x == 0x4:
            channel.vibrato_waveform = y
        elif x == 0x7:
            channel.tremolo_waveform = y

    # -- sample export ------------------------------------------------------

    def _sample_rate(self, asset: SampleAsset) -> float:
        sample = asset.sample
        if asset.fmt == "mod":
            period = MODSong.INV_PERIOD_TABLE["C-5"]
            base_freq = MODSong.PAL_CLOCK / (period * 2.0)
            finetune = int(getattr(sample, "finetune", 0))
            if finetune > 7:
                finetune -= 16
            return base_freq * (2.0 ** (finetune / 96.0))
        if asset.fmt == "xm":
            relative = int(getattr(sample, "relative_note", 0))
            finetune = int(getattr(sample, "finetune", 0))
            return 8363.0 * (2.0 ** ((relative + finetune / 128.0) / 12.0))
        # S3M c2spd is the playback rate of the C-5 reference note.
        rate = int(getattr(sample, "c2spd", 0) or 8363)
        return float(rate)

    @staticmethod
    def _is_16bit(asset: SampleAsset) -> bool:
        return bool(getattr(asset.sample, "is_16bit", False))

    def _export_samples(self):
        used = [
            asset
            for asset in sorted(self.assets.values(), key=lambda a: a.key)
            if self.segments_by_key.get(asset.key)
        ]
        self.assets_by_key = used
        for index, asset in enumerate(used):
            asset.filename = f"sample{index}.wav"
            asset.track_name = self._unique_track_name(asset.name, index)
            path = os.path.join(self.samples_dir, asset.filename)
            self._write_wav(asset, path)

    def _unique_track_name(self, name: str, index: int) -> str:
        cleaned = " ".join((name or "").split())
        if not cleaned:
            cleaned = f"sample {index}"
        candidate = cleaned
        suffix = 2
        while candidate.casefold() in self.used_track_names:
            candidate = f"{cleaned} {suffix}"
            suffix += 1
        self.used_track_names.add(candidate.casefold())
        return candidate

    def _write_wav(self, asset: SampleAsset, path: str):
        sample = asset.sample
        waveform = getattr(sample, "waveform", array("b"))
        is_16 = self._is_16bit(asset)
        rate = int(round(self._sample_rate(asset)))
        rate = int(clamp(rate, 1000, 192000))

        if is_16:
            data = array("h", waveform)
            if sys.byteorder != "little":
                data.byteswap()
            payload = data.tobytes()
        else:
            payload = bytes((int(value) + 128) & 0xFF for value in waveform)

        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2 if is_16 else 1)
            wav.setframerate(rate)
            wav.writeframes(payload)

    # -- LMMS project -------------------------------------------------------

    def _track_loop_values(self, asset: SampleAsset):
        if asset.loop_length <= 1 or asset.length <= 0:
            return 0, 0.0, 0.0, 1.0
        loop_mode = int(getattr(asset.sample, "loop_type", 1) or 1)
        if loop_mode not in (1, 2):
            loop_mode = 1
        start = asset.loop_start / asset.length
        loop = asset.loop_start / asset.length
        end = min(1.0, (asset.loop_start + asset.loop_length) / asset.length)
        return loop_mode, start, loop, end

    def _write_project(self):
        max_end = 0
        for segments in self.segments_by_key.values():
            for segment in segments:
                max_end = max(max_end, segment.end_tick or 0)
        clip_len = max(TICKS_PER_BAR, int(math.ceil(max_end / TICKS_PER_BAR) * TICKS_PER_BAR))

        root = ET.Element(
            "lmms-project",
            {
                "version": "31",
                "type": "song",
                "creatorversion": "1.3.0-alpha.2",
                "creator": "LMMS",
                "creatorplatform": "darwin",
                "creatorplatformtype": "osx",
            },
        )
        ET.SubElement(
            root,
            "head",
            {
                "masterpitch": "0",
                "bpm": fmt_number(self.base_bpm),
                "timesig_numerator": "4",
                "timesig_denominator": "4",
                "mastervol": "100",
            },
        )
        song_el = ET.SubElement(root, "song")
        container = ET.SubElement(
            song_el,
            "trackcontainer",
            {
                "type": "song",
                "height": "300",
                "maximized": "0",
                "width": "720",
                "x": "6",
                "y": "5",
                "visible": "1",
            },
        )

        for track_index, asset in enumerate(self.assets_by_key):
            self._add_track(container, asset, track_index, clip_len)

        self._add_song_tail(song_el, clip_len)

        indent = getattr(ET, "indent", None)
        if indent is not None:
            indent(root, space="  ")
        xml = ET.tostring(root, encoding="unicode")
        xml = '<?xml version="1.0"?>\n<!DOCTYPE lmms-project>\n' + xml + "\n"
        with open(os.path.join(self.output_dir, "song.mmp"), "w", encoding="utf-8") as handle:
            handle.write(xml)

    def _add_track(self, container, asset: SampleAsset, track_index: int, clip_len: int):
        loop_mode, sframe, lframe, eframe = self._track_loop_values(asset)
        track = ET.SubElement(
            container,
            "track",
            {
                "type": "0",
                "name": asset.track_name,
                "muted": "0",
                "mutedBeforeSolo": "1",
                "solo": "0",
            },
        )
        instrument_track = ET.SubElement(
            track,
            "instrumenttrack",
            {
                "keymap": "0",
                # Leave a little headroom for the summed tracker channels.
                "vol": "80",
                "lastkey": "127",
                "mixch": "0",
                "basenote": str(ROOT_KEY),
                "firstkey": "0",
                "enablecc": "0",
                "pitch": "0",
                "usemasterpitch": "1",
                "enabled": "0",
                "scale": "0",
                "range_import": "1",
                "pan": "0",
                "pitchrange": "1",
            },
        )
        controllers = ET.SubElement(instrument_track, "midicontrollers")
        for cc in range(128):
            controllers.set(f"cc{cc}", "0")

        instrument = ET.SubElement(instrument_track, "instrument", {"name": "audiofileprocessor"})
        audio = ET.SubElement(
            instrument,
            "audiofileprocessor",
            {
                "interp": "1",
                # "local:" tells LMMS to resolve the path relative to the
                # directory containing song.mmp.
                "src": f"local:samples/{asset.filename}",
                "looped": str(loop_mode),
                "sframe": fmt_number(sframe),
                "lframe": fmt_number(lframe),
                "stutter": "0",
                "amp": "100",
                "eframe": fmt_number(eframe),
                "reversed": "0",
            },
        )
        ET.SubElement(audio, "key")

        self._add_envelope_defaults(instrument_track)
        ET.SubElement(instrument_track, "chordcreator", {"chord": "0", "chord-enabled": "0", "chordrange": "1"})
        ET.SubElement(
            instrument_track,
            "arpeggiator",
            {
                "arpcycle": "0",
                "arpdir": "0",
                "arp-enabled": "0",
                "arprange": "1",
                "arptime_numerator": "4",
                "arpmiss": "0",
                "arpmode": "0",
                "arpskip": "0",
                "arptime": "200",
                "arptime_syncmode": "0",
                "arprepeats": "1",
                "arpgate": "100",
                "arptime_denominator": "4",
            },
        )
        ET.SubElement(
            instrument_track,
            "midiport",
            {
                "outputchannel": "1",
                "outputcontroller": "-1",
                "outputprogram": "1",
                "readable": "0",
                "fixedoutputvelocity": "-1",
                "writable": "0",
                "fixedinputvelocity": "-1",
                "basevelocity": "63",
                "fixedoutputnote": "-1",
                "inputchannel": "0",
                "inputcontroller": "-1",
            },
        )
        ET.SubElement(instrument_track, "fxchain", {"enabled": "0", "numofeffects": "0"})

        clip = ET.SubElement(
            track,
            "midiclip",
            {
                "type": "1",
                "name": "",
                "pos": "0",
                "muted": "0",
                "steps": "16",
                "off": "0",
                "len": str(clip_len),
                "autoresize": "1",
            },
        )
        for segment in sorted(
            self.segments_by_key.get(asset.key, []),
            key=lambda seg: (seg.start_tick, seg.key),
        ):
            self._add_note(clip, segment)

    def _add_envelope_defaults(self, parent):
        eldata = ET.SubElement(
            parent,
            "eldata",
            {"fwet": "0", "fres": "0.5", "ftype": "0", "fcut": "14000"},
        )
        for name in ("elvol", "elcut", "elres"):
            ET.SubElement(
                eldata,
                name,
                {
                    "dec": "0.5",
                    "att": "0",
                    "lspd": "0.1",
                    "pdel": "0",
                    "sustain": "0.5",
                    "amt": "0",
                    "hold": "0.5",
                    "ctlenvamt": "0",
                    "lspd_numerator": "4",
                    "latt": "0",
                    "userwavefile": "",
                    "lpdel": "0",
                    "lamt": "0",
                    "lshp": "0",
                    "lspd_syncmode": "0",
                    "lspd_denominator": "4",
                    "x100": "0",
                    "rel": "0.1",
                },
            )

    def _add_note(self, clip, segment: Segment):
        note = ET.SubElement(
            clip,
            "note",
            {
                "type": "0",
                "vol": str(volume_to_lmms(segment.volume)),
                "pos": str(segment.start_tick),
                "pan": str(segment.pan),
                "len": str(segment.length),
                "key": str(segment.key),
            },
        )
        if not segment.detune:
            return
        if all(abs(value) < 0.005 for _, value in segment.detune):
            return

        automated = ET.SubElement(note, "automationclip")
        length = max(segment.length, segment.detune[-1][0] + 1)
        detuning = ET.SubElement(
            automated,
            "detuning",
            {
                "mute": "0",
                "pos": "0",
                "name": "Note detuning",
                "tens": "1",
                "off": "0",
                "len": str(length),
                "autoresize": "1",
                "prog": "1",
            },
        )
        for pos, value in segment.detune:
            ET.SubElement(
                detuning,
                "time",
                {
                    "value": fmt_number(value),
                    "pos": str(pos),
                    "outValue": fmt_number(value),
                    "outTan": "0",
                    "lockedTan": "0",
                    "inTan": "0",
                },
            )

    def _add_song_tail(self, song_el, clip_len):
        mixer = ET.SubElement(
            song_el,
            "mixer",
            {
                "height": "385",
                "maximized": "0",
                "width": "607",
                "x": "5",
                "y": "310",
                "visible": "1",
            },
        )
        master = ET.SubElement(
            mixer,
            "mixerchannel",
            {
                "name": "Master",
                "muted": "0",
                "volume": "1",
                "soloed": "0",
                "num": "0",
            },
        )
        ET.SubElement(master, "fxchain", {"enabled": "0", "numofeffects": "0"})

        ET.SubElement(
            song_el,
            "ControllerRackView",
            {
                "height": "228",
                "maximized": "0",
                "width": "358",
                "x": "836",
                "y": "407",
                "visible": "0",
            },
        )
        ET.SubElement(
            song_el,
            "pianoroll",
            {
                "height": "480",
                "maximized": "1",
                "width": "970",
                "x": "1",
                "scale": "0",
                "y": "1",
                "chord": "0",
                "stopbehaviour": "0",
                "visible": "0",
                "key": "0",
            },
        )
        ET.SubElement(
            song_el,
            "automationeditor",
            {
                "height": "400",
                "maximized": "1",
                "width": "860",
                "x": "56",
                "y": "255",
                "visible": "0",
            },
        )
        notes = ET.SubElement(
            song_el,
            "projectnotes",
            {
                "height": "400",
                "maximized": "0",
                "width": "742",
                "x": "1",
                "y": "1",
                "visible": "0",
            },
        )
        notes.text = (
            "<html><body><p>Converted from "
            + os.path.basename(self.module_path)
            + " by modtommp.py</p></body></html>"
        )
        ET.SubElement(
            song_el,
            "timeline",
            {"lp0pos": "0", "lpstate": "0", "stopbehaviour": "1", "lp1pos": str(clip_len)},
        )
        ET.SubElement(song_el, "controllers")
        scales = ET.SubElement(song_el, "scales")
        for _ in range(10):
            scale = ET.SubElement(scales, "scale", {"description": "empty"})
            ET.SubElement(scale, "interval", {"den": "1", "num": "1"})
        keymaps = ET.SubElement(song_el, "keymaps")
        for _ in range(10):
            ET.SubElement(
                keymaps,
                "keymap",
                {
                    "last_key": "127",
                    "description": "empty",
                    "first_key": "0",
                    "middle_key": "60",
                    "base_freq": "440",
                    "base_key": "69",
                },
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _ask_value(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        value = ""
    return value or default


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print("Usage: modtommp.py [module] [output-folder]")
        print()
        print("Without arguments, the program asks for the module file and folder.")
        print("It writes <folder>/song.mmp and <folder>/samples/sampleN.wav.")
        print("Supported formats: MOD (.mod), XM (.xm), S3M (.s3m).")
        return 0

    if argv:
        module_path = argv[0]
    else:
        module_path = _ask_value("Song filename")

    if not module_path:
        print("No module file given.", file=sys.stderr)
        return 2
    if not os.path.isfile(module_path):
        print(f"File not found: {module_path}", file=sys.stderr)
        return 2

    stem = os.path.splitext(os.path.basename(module_path))[0]
    default_folder = f"{stem}_mmp"
    if len(argv) > 1:
        output_dir = argv[1]
    else:
        output_dir = _ask_value("Output folder", default_folder)

    if not output_dir:
        print("No output folder given.", file=sys.stderr)
        return 2

    print(f"Loading {module_path}...")
    try:
        song = load_song(module_path, verbose=False)
    except Exception as exc:  # noqa: BLE001 - report a helpful CLI error
        print(f"Could not load {module_path}: {exc}", file=sys.stderr)
        print(
            "modtommp currently reads the formats supported by nodmod: "
            "MOD (.mod), XM (.xm) and S3M (.s3m).",
            file=sys.stderr,
        )
        return 1

    print(
        f"Loaded {song.file_extension.upper()} module "
        f"'{getattr(song, 'songname', stem)}' "
        f"with {song.n_channels} channels."
    )

    converter = MMPConverter(song, module_path, output_dir)
    converter.run()

    print(f"Wrote {os.path.join(output_dir, 'song.mmp')}")
    print(f"Wrote {len(converter.assets_by_key)} sample(s) to {converter.samples_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
