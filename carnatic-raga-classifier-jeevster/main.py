import yaml
from types import SimpleNamespace
from inference import Evaluator
import librosa


with open("config0.yaml") as stream:
    params = yaml.safe_load(stream)

params = SimpleNamespace(**params)
params.device="cpu"
params.input_channels=2

ev = Evaluator(params)


data_dir = "/Users/neerajaabhyankar/Repos/icm-shruti-analysis/data-dunya-hindustani/"
track = "Omkar Dadarkar - Raag Bhoopali"
y, sr = librosa.load(data_dir + track + ".mp3")

result = ev.inference(10, (y, sr))
