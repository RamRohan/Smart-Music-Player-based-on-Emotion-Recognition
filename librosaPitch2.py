'''
import librosa

print 'x'
y, sr = librosa.load('Therapy.mp3')
pitches, magnitude = librosa.core.piptrack(y=y, sr=sr)
    #final = sum(pitches)/len(pitches)
print pitches, magnitude
'''   


"""
Evaluate Google's REAPER, a pitch detector (https://github.com/google/REAPER), on a bunch of audio files.
 
The audio files must have ground-truth .lab annotations on the format:
 
48  1.104   1.573
50  1.623   1.715
52  1.754   2.055
53  2.094   2.367
55  2.412   2.728
 
where the first column is the MIDI note number, the second column is the note onset time in seconds, and the third and final column is the note offset time in seconds. Each row represents one note. Notes don't have to be sorted in any particular order. The values are tab-delimited.
"""
'''
import argparse
import glob
import os
import subprocess
 
import librosa as lr
import mir_eval as me
import numpy as np
import pandas as pd
 
parser = argparse.ArgumentParser()
parser.add_argument(
    'directory',
    help="directory with audio files and ground-truth annotations", )
parser.add_argument(
    'reaper',
    help="full path to REAPER binary", )
args = parser.parse_args()
directory = args.directory
reaper = args.reaper
 
for f in glob.glob(os.path.join(directory, '*.wav')):
 
    # Load audio.
    name, extension = os.path.splitext(f)
    y, sr = lr.load(f)
 
    # Estimate pitch with REAPER.
    if not os.path.isfile(name + '.f0'):
        subprocess.call([reaper, '-i', f, '-f', name + '.f0', '-a'])
    #pitches, magnitudes = lr.piptrack(y, sr, fmin=80, fmax=250)  # TODO Compare with libROSA.
    df = pd.read_csv(
        name + '.f0',
        skiprows=np.arange(7),
        names=['time', 'on', 'pitch'],
        delimiter=' ', )
    times = df['time']
    pitches = df['pitch']
 
    # Filter by both REAPER VAD and libROSA's onset detection.
    dt = 0.03
    keep = np.zeros_like(df['on'])
    for t in lr.frames_to_time(lr.onset.onset_detect(y, sr), sr):
        keep[times[np.abs(times - t) < dt].index] = True
    times = df['time'][np.logical_and(keep, df['on'])]
    pitches = df['pitch'][np.logical_and(keep, df['on'])]
 
    # Save annotation.
    lr.output.times_csv(name + '.csv', times, pitches)
 
    # Evaluate annotation.
    estimate = pd.read_csv(
        name + '.csv',
        names=['onset', 'pitch'], )
    reference = pd.read_csv(
        name + '.lab',
        delimiter='\t',
        names=['note', 'onset', 'offset'], )
    evaluation = me.transcription.precision_recall_f1_overlap(
        ref_intervals=reference[['onset', 'offset']].as_matrix(),
        ref_pitches=me.util.midi_to_hz(reference['note']).as_matrix(),
        est_intervals=np.vstack((estimate['onset'].as_matrix(),
                                 (estimate['onset'] + 0.5).as_matrix())).T,
        est_pitches=me.util.midi_to_hz(
            np.round(me.util.hz_to_midi(estimate['pitch']))).as_matrix(),
        offset_ratio=None)
    print(evaluation)
'''
import librosa
import librosa.display
from matplotlib import pyplot as plt
import numpy as np
import math

def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    
    # Compute autocorrelation of input segment.
    print '1'
    r = librosa.autocorrelate(segment)
    print '1'
    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    print '1'
    i_max = sr/fmin
    print '1'
    r[:int(i_min)] = 0
    print '1'
    r[int(i_max):] = 0
    print '1'
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    print '1'
    f0 = float(sr)/i
    print '1'
    return f0

y, sr = librosa.load('Imagine Dragons - Demons.mp3')

librosa.feature.chroma_stft(y=y, sr=sr)
S = np.abs(librosa.stft(y))
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

sum1 = []
i = 0
for x in chroma:
    sum2=0
    i = i+1
    for y in x:
        sum2=sum2+y
    sum2 = sum2/len(x)
    sum1 = sum1 + [sum2]

#print 'n_chroma :', i
maxi = sum1[0]
pos = 0
for i in range(0,12):
    if sum1[i]>maxi:
        maxi = sum1[i]
        pos = i

if(pos>6):
        pos = pos - 6
else :
        pos = pos + 6

print pos

'''
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()
'''
'''

bins_per_octave = 36
cqt = librosa.cqt(y, sr=sr, n_bins=300, bins_per_octave=bins_per_octave)
log_cqt = librosa.logamplitude(cqt)

hop_length = 100
onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length)

plt.plot(onset_env)
plt.xlim(0, len(onset_env))
plt.show()


onset_samples = librosa.onset.onset_detect(y,
                                           sr=sr, units='samples', 
                                           hop_length=hop_length, 
                                           backtrack=False,
                                           pre_max=20,
                                           post_max=20,
                                           pre_avg=100,
                                           post_avg=100,
                                           delta=0.2,
                                           wait=0)
onset_boundaries = numpy.concatenate([[0], onset_samples, [len(y)]])
onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)


librosa.display.waveplot(y, sr=sr)
plt.vlines(onset_times, -1, 1, color='r')
plt.show()
'''
#f0 = estimate_pitch(y[0:len(y)], sr)
#print f0
#f0 = 2004.54545455
'''
p = math.log((f0/440), 2)
final = 69 + 12*p
print final
'''
