import pretty_midi
import numpy as np
from pathlib import Path

def to_midi(generated, out_path='gen.mid', program=0, vel_norm=True, min_dur=0.02):
    """
    generated: list of (pitch, vel, dur) where vel may be None or normalized in [0,1]
    out_path: path to save .mid
    program: MIDI program (instrument) number, default 0 = Acoustic Grand Piano
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    current_time = 0.0

    for (pitch, vel, dur) in generated:
        # Skip invalid or rest tokens (use None or negative pitch to indicate rest)
        if pitch is None or pitch < 0:
            current_time += max(min_dur, float(dur) if dur is not None else min_dur)
            continue

        # Clamp pitch
        pitch = int(max(0, min(127, int(round(pitch)))))

        # Velocity: denormalize from [0,1] -> 0..127 if vel_norm else assume already 0..127
        if vel is None:
            vel_int = 64
        else:
            vel_val = float(vel)
            if vel_norm:
                vel_int = int(round(max(0, min(1.0, vel_val)) * 127))
            else:
                vel_int = int(round(max(0, min(127, vel_val))))

        # Duration
        dur_val = float(dur) if dur is not None else min_dur
        dur_val = max(min_dur, dur_val)

        start = current_time
        end = current_time + dur_val

        note = pretty_midi.Note(velocity=vel_int, pitch=pitch, start=start, end=end)
        inst.notes.append(note)

        # simple cursor advance (monophonic)
        current_time = end

    pm.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(out_path)
    return out_path