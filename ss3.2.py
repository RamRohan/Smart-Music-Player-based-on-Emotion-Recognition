import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib2
import json
from textblob import TextBlob
import re
#import librosa
import subprocess
import glob
import os
import sklearn.metrics
from sklearn.metrics import silhouette_score
import MySQLdb
import signal

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd=" ",  # your password
                     db="MUSIC")        # name of the data base

emotion= "Happy"

pl=input("Enter how important lyrics is to you on a scale of one to ten: ")#percentage for lyrics
pi=input("Enter how important the instrumental is to you on a scale of one to ten: ")#percentage for instrumetal
#%matplotlib inline

cur = db.cursor()
cur.execute('SELECT NAME, AROUSAL, VALENCE FROM SONGS')
data = cur.fetchall()
songs = []
valence = []
arousal = []
emotions = []
for x in data:
  songs = songs + [x[0]]
  arousal = arousal + [x[1]]
  valence = valence + [x[2]]
  emotions = emotions + ['NULL']
'''
songs = ['AllTimeLow-StayAwake.mp3', 'Evanescence-BringMetoLife.mp3', 'BrittNicole-TheLostGetFound.mp3', 'CharityVance-IntoTheOcean.mp3', 'CD9-QuimicaEnComun.mp3', 'OneDirection-Moments.mp3', 'AlanWalker-Faded.mp3', 'BeaMiller-OpenYourEyes(DeepBlueSongspell).mp3', 'Oh,Come,AllYeFaithfulMusicVideoft.BYUVocalPointandBYUNoteworthy.mp3', 'AgainstTheCurrent-Wasteland.mp3', '5SecondsofSummer-Amnesia.mp3', 'ShowMetheMeaningofBeingLonely-(www.SongsLover.com).mp3', 'AllTimeLow-Backseatserenade.mp3', 'OneDirection-NaNaNa.mp3', 'CharityVance-HelloChange.mp3', 'BackstreetBoys-Aslongyouloveme.mp3', 'TaylorSwift-Red.mp3', 'AgainstTheCurrent-ForgetMeNow.mp3', 'ChristinaGrimmieandDiamondEyes-StayWithMe.mp3', 'ImagineDragons-Demons.mp3', 'EdSheeran-TheATeam.mp3', 'AdamLambert-BetterThanIKnowMyself.mp3', 'OneDirection-SameMistakes.mp3', 'OneDirection-Tellmealie.mp3', 'BackstreetBoys-Drowning.mp3', 'AllTimeLow-OldScarsFutureHearts.mp3', 'BackstreetBoys-IWantItThatWay.mp3', 'CD9-Termino(O.M.G.).mp3', 'OneDirection-Taken.mp3', 'DemiLovato-Nightingale.mp3', '5SecondsOfSummer-GreenLight.mp3', 'CelineDion-MyHeartWillGoOn.mp3']
valence = [0.8333333333333334, 0.16666666666666666, 0.16666666666666666, 0.25, 0.4166666666666667, 0.0, 0.08333333333333333, 0.5833333333333334, 0.5833333333333334, 0.0, 0.75, 0.9166666666666666, 0.3333333333333333, 0.9166666666666666, 0.0, 0.25, 0.16666666666666666, 0.4166666666666667, 0.25, 0.5, 0.0, 0.08333333333333333, 0.9166666666666666, 0.8333333333333334, 0.16666666666666666, 0.0, 0.75, 0.16666666666666666, 0.4166666666666667, 0.0, 0.4166666666666667, 0.5833333333333334]
arousal = [0.79133997999427474, 0.47857177108310645, 0.21981362134216345, 0.54132751457986927, 0.83781864386243943, 0.4682279975690572, 0.51417953922446291, 0.41561340423960264, 0.27522091799367188, 0.55986548448206508, 0.61233680028455084, 0.31863807255022819, 0.68701897605045703, 0.38831501963650927, 0.56703572359285881, 0.37927804259760012, 0.55744382136309101, 0.61314738031382432, 0.69889259500762724, 0.54032004887150165, 0.26818466525708462, 0.74481787730850424, 0.47916313052005832, 0.6535842173965194, 0.5317892058453223, 0.70832744011232518, 0.36382384480442498, 0.83311637194773802, 0.45897919739917287, 0.58397457698758348, 0.74758106591704809, 0.21886849312249831]
'''
def get_lyrval(song):
  api_key="db4950ba602b5cd2e06b3f2b151ff6cc"
  url="http://api.musixmatch.com/ws/1.1/matcher.lyrics.get?q_track={}&apikey={}".format(song,api_key)
  req = urllib2.Request(url)
  req.add_header("Accept", "application/json")
  response = urllib2.urlopen(req).read()
  data=json.loads(response)
  if(data["message"]["header"]["status_code"]==200):
    #print y
    lyrics=data["message"]["body"]["lyrics"]["lyrics_body"]
    print(lyrics)
    test= TextBlob(lyrics)
    x=test.sentiment.polarity
    return ((x+1)/2)
  else:
    return -1

i=0
for sng in songs:
   sng=sng.replace(".mp3","")
   #text=text.split(spl,1)[0]
   sng=re.sub("[\(\[].*?[\)\]]", "", sng)
   sng=sng.replace(" ","%20")
   val=get_lyrval(sng)
   if (val!=-1):
      valence[i]=(pi*valence[i]+pl*val)/(pl+pi)
   i=i+1

#print (valence)

df = pd.DataFrame({
    'x': valence,
    'y': arousal
})


#np.random.seed(200)
k = 4
# centroids[i] = [x, y]
'''
#centroids = {
 #   i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
  #  for i in range(k)
#}
'''
centroids={1:[0.8,0.8], 2:[0.2,0.8], 3:[0.2,0.2], 4:[0.8,0.2]}
#centroids={1:[1,1], 2:[0,1], 3:[0,1], 4:[1,0]}
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b', 4:'y'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0.00,1.00)
plt.ylim(0.00,1.00)
plt.title = '1'
#plt.show()

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
#print(df.head())

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,1)
plt.ylim(0,1)
plt.title = '2'
#plt.show()

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,1)
plt.ylim(0,1)
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=0.05, head_length=0.075, fc=colmap[i], ec=colmap[i])
plt.title = '3'
#plt.show()
df = assignment(df, centroids)
# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,1)
plt.ylim(0,1)
plt.title = '4'
#plt.show()

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,1)
plt.ylim(0,1)
plt.title = '5'
plt.show()            

print("The four clusters have the following valence and arousal values:")
for i in centroids.keys():
    print("Emotion number {} - Valence: {}".format(i,centroids[i][0]))
    print("                  - Arousal: {}".format(centroids[i][1]))

#print("Emotion 1: Red(r) Emotion 2: Green(g) Emotion 3: Blue(b) Emotion 4: Yellow(y)")
i=0
print("Emotion 1- delighted/happy")
for a in df['color']:
   if a=='r':
      print(songs[i])
      emotions[i] = 'HAPPY'
   i=i+1
i=0
print ''
print ''
print("Emotion 2- angry/annoyed")
for a in df['color']:
   if a=='g':
      print(songs[i])
      emotions[i] = 'ANGRY'
   i=i+1
i=0
print ''
print ''
print("Emotion 3- depressed/sad")
for a in df['color']:
   if a=='b':
      print(songs[i])
      emotions[i] = 'SAD'
   i=i+1
i=0
print ''
print ''
print("Emotion 4- calm/relaxed")
for a in df['color']:
   if a=='y':
      print(songs[i])
      emotions[i] = 'CALM'
   i=i+1
i = 0
for x in data:
  cur.execute('UPDATE SONGS SET EMOTION = %s, VALENCE = %s, AROUSAL = %s WHERE NAME = %s', (emotions[i], valence[i], arousal[i], songs[i],))
  db.commit()
  i = i+1
y=[]
np.random.seed(0)
for i in range(0,len(valence)):
  y.append([valence[i],arousal[i]])

l=df['closest']
avg = silhouette_score(y, l)
print avg


