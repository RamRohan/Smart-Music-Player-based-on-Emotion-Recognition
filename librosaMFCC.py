import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('Therapy.mp3')

mfcc = librosa.feature.mfcc(y=y, sr=sr)

#print mfcc

#Let's scale the MFCCs such that each coefficient dimension has zero mean and unit variance
mfccs = sklearn.preprocessing.scale(mfcc, axis=1)
print mfccs.mean(axis=1)
print mfccs.var(axis=1)


plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
