import matplotlib.pyplot as plt
import librosa
import librosa.display


y, sr = librosa.load(librosa.util.example_audio_file())
librosa.feature.zero_crossing_rate(y)
