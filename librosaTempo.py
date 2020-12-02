import numpy as np
import librosa
from matplotlib import pyplot as plt

y,sr = librosa.load('Therapy.mp3')
#spectrogram = np.abs(librosa.stft(y))
tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
#beat_times = librosa.frames_to_time(frames, sr=sr)

#print beat_times

print tempo
