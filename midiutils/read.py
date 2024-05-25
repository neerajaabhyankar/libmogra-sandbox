import mido
import librosa
import librosa.display
import numpy as np
from typing import Dict, List


SR = 22050  # Sampling rate for audio
VELOCITY_SCALING = 127.0  # Maximum MIDI velocity
NOTE_ON_VELOCITY_THRESHOLD = 0  # Threshold to consider a note_on event as actually on


def midi_to_freq(midi_note):
    """ Helper function to convert MIDI note to frequency
    """
    return 440.0 * 2**((midi_note - 69) / 12.0)


def to_pitch_class(note, root):
    return librosa.midi_to_svara_h(note, Sa=root, abbr=True, octave=False, unicode=False)


def parse_file(filename) -> Dict:
    """ Return track of notes, length, ticks per beat
    """
    mid = mido.MidiFile(filename)
    
    track = mid.tracks[0]

    # # meta
    # for ii, mm in enumerate(track):
    #     if type(mm) != mido.messages.messages.Message:
    #         print(mm)

    # # other signals
    # for ii, mm in enumerate(track):
    #     if type(mm) == mido.messages.messages.Message:
    #         if mm.type != "note_on":
    #             print(mm)

    track_notes = []
    for ii, mm in enumerate(track):
        if type(mm) == mido.messages.messages.Message:
            if mm.type == "note_on":
                track_notes.append(mm)

    return {
        "notes": track_notes,
        "length": mid.length,
        "ticks_per_beat": mid.ticks_per_beat
    }


def get_audio(filename):
    parsed = parse_file(filename)
    track_notes = parsed["notes"]
    midi_length = parsed["length"]
    midi_ticks_per_beat = parsed["ticks_per_beat"]

    # Create a blank audio array
    audio_length = int(mido.tick2second(midi_length, midi_ticks_per_beat, 500000) * SR)
    audio = np.zeros(audio_length*SR)
    # Create a blank symbolic array
    # TODO(neeraja): take notes equi-spaced in time
    nts = np.array([0,]*len(track_notes))


    # Time tracking
    current_time = 0

    for ii, msg in enumerate(track_notes):
        assert msg.type == "note_on"
        nts[ii] = msg.note
        
        # Calculate time in seconds and samples
        note_time = mido.tick2second(msg.time, midi_ticks_per_beat, 500000)
        note_samples = int(note_time * SR)

        # Calculate the start and end sample indices
        start_sample = current_time
        end_sample = current_time + note_samples

        if msg.velocity >= NOTE_ON_VELOCITY_THRESHOLD:  # Note on event
            freq = midi_to_freq(msg.note)
            duration = (note_samples / SR)

            # Generate a sine wave for the note
            t = np.linspace(0, duration, note_samples, False)
            wave = 0.5 * np.sin(2 * np.pi * freq * t) * (msg.velocity / VELOCITY_SCALING)

            # Add the generated wave to the audio array
            assert len(wave) == end_sample - start_sample
            audio[start_sample:start_sample+len(wave)] += wave

        # Move current time forward
        current_time = end_sample

    # Normalize the audio to avoid clipping
    audio = audio / np.max(np.abs(audio))
    
    return audio, SR


def get_symbol_string(filename, root):
    parsed = parse_file(filename)
    track_notes = parsed["notes"]
    
    # TODO(neeraja): take notes equi-spaced in time
    nts = np.array([0,]*len(track_notes))

    for ii, msg in enumerate(track_notes):
        assert msg.type == "note_on"
        # if msg.velocity >= NOTE_ON_VELOCITY_THRESHOLD:
        nts[ii] = msg.note

    syms = [to_pitch_class(nn, root) for nn in nts]
    
    return syms
