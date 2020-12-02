from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from matplotlib import pyplot as plt
import librosa

#import sys, aubio, subprocess

samplerate = 0  # use original source samplerate
hop_size = 256 # number of frames to read in one block
#subprocess.call(['ffmpeg', '-i', sys.argv[1],'xfile.wav'])

[Fs, x] = audioBasicIO.readAudioFile('xfile.wav');
print Fs, x
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);

print F

plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel('ZCR'); 
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy');
plt.show();
