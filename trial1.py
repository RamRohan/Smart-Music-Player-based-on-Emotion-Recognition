import numpy as np
import librosa
from matplotlib import pyplot as plt

y,sr = librosa.load('01. Still 24K.mp3')
tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
print tempo
