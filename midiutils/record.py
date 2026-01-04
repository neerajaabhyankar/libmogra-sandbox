import mido
import time
import argparse


def record_session(midi_outfile, text_outfile):
    inputs = mido.get_input_names()
    selected_input = "Minilab3 MIDI" if "Minilab3 MIDI" in inputs else inputs[0]
    print("Inputs:", inputs)
    print("Selected:", selected_input)
    inport = mido.open_input(selected_input)

    t0 = time.perf_counter()

    # midi file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    last_t = 0
    # text file
    fp = open(text_outfile, "w")

    try:
        for msg in inport:
            t = time.perf_counter() - t0
            if msg.type in ("note_on", "note_off"):
                delta = t - last_t
                last_t = t
                ticks = int(mido.second2tick(delta, mid.ticks_per_beat, 500000))
                track.append(msg.copy(time=ticks))
                print(t, msg)
            if msg.type == "note_on":
                fp.write(f"{t:.3f} {msg.note}\n")
            if msg.type == "note_off":
                fp.write(f"{t:.3f} {-1}\n")
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    
    finally:
        inport.close()
        mid.save(midi_outfile)
        fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outprefix", type=str, default="out")
    args = parser.parse_args()
    
    outprefix = args.outprefix.split(".")[0]
    record_session(outprefix + "_raw.mid", outprefix + "_raw.txt")