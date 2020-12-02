import numpy as np
import librosa
from matplotlib import pyplot as plt

'''
#filename = librosa.util.example_audio_file()
#y,sr = librosa.load(filename)#, offset = 25.0, duration = 20.0)
y,sr = librosa.load('Therapy.mp3')
#spectrogram = np.abs(librosa.stft(y))
tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
#beat_times = librosa.frames_to_time(frames, sr=sr)

#print beat_times

print tempo
#print spectrogram

#librosa.display.waveplot(y)
'''
'''
y, sr = librosa.load(librosa.util.example_audio_file())
print(len(y), sr)

y_orig, sr_orig = librosa.load(librosa.util.example_audio_file(),
                     sr=None)
print(len(y_orig), sr_orig)

sr = 22050

y = librosa.resample(y_orig, sr_orig, sr)

print(len(y), sr)
print(librosa.samples_to_time(len(y), sr))

D = librosa.stft(y)
print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
print(S.dtype, phase.dtype, np.allclose(D, S * phase))

C = librosa.cqt(y, sr=sr)

print(C.shape, C.dtype)

y2, sr2 = librosa.load('Therapy.mp3')
D = librosa.stft(y2)
print(D.shape, D.dtype)
D = librosa.stft(y2, hop_length=2)
print(D.shape, D.dtype)
'''

y,sr = librosa.load('Therapy.mp3')
#spectrogram = np.abs(librosa.stft(y))
tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
#beat_times = librosa.frames_to_time(frames, sr=sr)

#print beat_times

print tempo
