import librosa
import librosa.display
y, sr = librosa.load('03 Melted.mp3')
rms = librosa.feature.rmse(y=y)
sum1 = 0
for i in range(0, len(rms[0])):
        sum1+=rms[0][i]**2

sum2 = sum1/len(rms[0])
print sum2
