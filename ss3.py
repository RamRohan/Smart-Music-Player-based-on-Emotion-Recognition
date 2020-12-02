import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib2
import json
from textblob import TextBlob
import re
import librosa
import subprocess
import glob
import os


pl=input("Enter how important lyrics is to you on a scale of one to ten: ")#percentage for lyrics
pi=input("Enter how important the instrumental is to you on a scale of one to ten: ")#percentage for instrumetal
#%matplotlib inline

'''
filepath = 'songs/*.mp3'
path = os.getcwd()
list2 = glob.glob(filepath)
for x in list2:
        r = x.replace("_", "")
        os.rename(x,r)

songs = []
valence = []
arousal = []

#Calculating arousal and valence for each song it has detected
for song in list2:
        
        y,sr = librosa.load(song)

        #tempo
        tempo, frames = librosa.beat.beat_track(y=y, sr=sr)
        ar1 = (tempo-40)/150

        #RMS energy
        rms = librosa.feature.rmse(y=y)
        sum1 = 0
        for i in range(0, len(rms[0])):
                sum1+=rms[0][i]**2

        ar2 = sum1/len(rms[0])
        ar2 = ar2/0.105

        ar = (ar1+ar2)/2

        #Chroma, Pitch classes
        librosa.feature.chroma_stft(y=y, sr=sr)
        S = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)

        sum1 = []
        for x in chroma:
            sum2=0
            for y in x:
                sum2=sum2+y
            sum2 = sum2/len(x)
            sum1 = sum1 + [sum2]

        maxi = sum1[0]
        pos = 0
        for i in range(0,12):
            if sum1[i]>maxi:
                maxi = sum1[i]
                pos = i

        if(pos>8):
                pos = pos - 9
        else :
                pos = pos + 3

        val1 = pos/12.0

        valence = valence + [val1]
        arousal = arousal + [ar]
        songs = songs + [song[6:]]

print 'songs =', songs
print 'valence =', valence
print 'arousal =', arousal


songs = ['All Time Low - Stay Awake.mp3', 'as long you love me.mp3', 'Red_ATC.mp3', 'backseat_serenade.mp3', 'Hello Change - Charity Vance_(new).mp3', 'Show Me the Meaning of Being Lonely-(www.SongsLover.com).mp3', '01. The A Team (www.SongsLover.com).mp3', 'Imagine Dragons - Demons.mp3', '1-09 Wasteland.mp3', '10 drowning.mp3', '06. Nightingale - (www.SongsLover.pk).mp3', '1-02 Forget Me Now.mp3', 'Amnesia (Lyric video)(1).mp3', 'I-want-it-that-way.mp3', '_ Bea Miller - Open Your Eyes (Deep Blue Songspell) (Audio Only).mp3', 'Into The Ocean - Charity Vance_(new).mp3', 'Green Light (Studio Version) - 5 Seconds Of Summer.mp3']
valence=[0.8333333333333334, 0.25, 0.16666666666666666, 0.3333333333333333, 0.0, 0.9166666666666666, 0.0, 0.5, 0.0, 0.16666666666666666, 0.0, 0.4166666666666667, 0.75, 0.75, 0.5833333333333334, 0.25, 0.4166666666666667]
arousal=[0.60657775707511918, 0.34365898318104776, 0.45494316246562272, 0.5383509780514899, 0.48881802246437589, 0.29980073909693433, 0.27539655863496942, 0.48286724440753837, 0.43846739017039182, 0.45685534869379418, 0.48425266854348137, 0.46367971841475775, 0.46899635139938922, 0.33554552933963078, 0.39586305285079137, 0.46819625581619606, 0.57014607523145022]

songs = ['Britt Nicole - The Lost Get Found.mp3', 'All Time Low - Stay Awake.mp3', 'Na Na Na - One direction lyric video with pictures.mp3', 'All Time Low - Old Scars_Future Hearts_(new).mp3', 'as long you love me.mp3', 'Red_ATC.mp3', 'Celine Dion-My Heart Will Go On.mp3', 'Stay With Me - Christina Grimmie _ Diamond Eyes (Official Lyric Video)_(new).mp3', "All Time Low - The Girl's A Straight-Up Hustler.mp3", '07 - Better Than I Know Myself - (www.SongsLover.pk).mp3', "08 We're All Thieves.mp3", 'All Time Low - Jasey Rae.mp3', 'backseat_serenade.mp3', 'same_mistakes.mp3', 'Hello Change - Charity Vance_(new).mp3', 'Tell_me_a_lie.mp3', 'CD9 - Termino (O.M.G.) (AparatajeMusic.NeT).mp3', 'Show Me the Meaning of Being Lonely-(www.SongsLover.com).mp3', '01. The A Team (www.SongsLover.com).mp3', 'Imagine Dragons - Demons.mp3', '1-09 Wasteland.mp3', '10 drowning.mp3', 'All Time Low - The Edge Of Tonight_(new).mp3', 'CD9 - Qu\xc3\xadmica En Com\xc3\xban (AparatajeMusic.NeT).mp3', 'One Direction- Moments [music video] [LoudTronix.me].mp3', '09 - Evanescence - Bring Me to Life.mp3', '06. Nightingale - (www.SongsLover.pk).mp3', 'Thanks To You.mp3', '1-02 Forget Me Now.mp3', 'Amnesia (Lyric video)(1).mp3', 'I-want-it-that-way.mp3', '_ Bea Miller - Open Your Eyes (Deep Blue Songspell) (Audio Only).mp3', 'Into The Ocean - Charity Vance_(new).mp3', 'Taken.mp3', 'Green Light (Studio Version) - 5 Seconds Of Summer.mp3']
valence = [0.16666666666666666, 0.8333333333333334, 0.9166666666666666, 0.0, 0.25, 0.16666666666666666, 0.5833333333333334, 0.25, 0.4166666666666667, 0.08333333333333333, 0.08333333333333333, 0.4166666666666667, 0.3333333333333333, 0.9166666666666666, 0.0, 0.8333333333333334, 0.16666666666666666, 0.9166666666666666, 0.0, 0.5, 0.0, 0.16666666666666666, 0.25, 0.4166666666666667, 0.0, 0.16666666666666666, 0.0, 0.4166666666666667, 0.4166666666666667, 0.75, 0.75, 0.5833333333333334, 0.25, 0.4166666666666667, 0.4166666666666667]
arousal = [0.22209010970093829, 0.81065714175440506, 0.40243454666000134, 0.72169954128460811, 0.38834460899030065, 0.57147486659791213, 0.21991458204144373, 0.71657811017467521, 0.3714570323929488, 0.76180893393434612, 0.55751592769496672, 0.40873031428891093, 0.7041108102696465, 0.49493746879893474, 0.57513767253291848, 0.67026356532774889, 0.85452235330554149, 0.327344253731503, 0.27390509643660554, 0.54529178048174332, 0.57796142296898889, 0.54111955155425506, 0.55828988016830317, 0.86245046147222792, 0.48174206171033052, 0.49321650547059503, 0.59591419125362932, 0.48924689840798885, 0.63452089516284882, 0.63239374446544505, 0.37211770130746669, 0.41913495986824945, 0.54972737682847062, 0.47264430310246486, 0.76770100462956714]

songs = ['5 Seconds Of Summer - Green Light.mp3', 'Britt Nicole - The Lost Get Found.mp3', 'One Direction - Same Mistakes.mp3', 'All Time Low - Stay Awake.mp3', '_ Bea Miller - Open Your Eyes (Deep Blue Songspell).mp3', 'Against The Current - Forget Me Now.mp3', 'One Direction - Taken.mp3', 'Taylor Swift -Red.mp3', 'Celine Dion-My Heart Will Go On.mp3', 'Backstreet Boys - As long you love me.mp3', 'Charity Vance - Into The Ocean.mp3', 'Ed Sheeran - The A Team.mp3', 'All Time Low - Old Scars Future Hearts.mp3', 'Backstreet Boys - Drowning.mp3', 'Christina Grimmie and Diamond Eyes - Stay With Me.mp3', 'Against The Current - Wasteland.mp3', 'Demi Lovato - Nightingale.mp3', 'Show Me the Meaning of Being Lonely-(www.SongsLover.com).mp3', 'Evanescence - Bring Me to Life.mp3', 'Imagine Dragons - Demons.mp3', 'One Direction- Moments.mp3', '5 Seconds of Summer -Amnesia.mp3', 'Backstreet Boys - I Want It That Way.mp3', 'CD9 - Termino (O.M.G.).mp3', 'CD9 - Quimica En Comun.mp3', 'One Direction - Na Na Na.mp3', 'Charity Vance - Hello Change.mp3', 'All Time Low - Backseat serenade.mp3', 'Adam Lambert - Better Than I Know Myself.mp3', 'One Direction - Tell me a lie.mp3']
valence = [0.4166666666666667, 0.16666666666666666, 0.9166666666666666, 0.8333333333333334, 0.5833333333333334, 0.4166666666666667, 0.4166666666666667, 0.16666666666666666, 0.5833333333333334, 0.25, 0.25, 0.0, 0.0, 0.16666666666666666, 0.25, 0.0, 0.0, 0.9166666666666666, 0.16666666666666666, 0.5, 0.0, 0.75, 0.75, 0.16666666666666666, 0.4166666666666667, 0.9166666666666666, 0.0, 0.3333333333333333, 0.08333333333333333, 0.8333333333333334]
arousal = [0.76770100462956714, 0.22209010970093829, 0.49493746879893474, 0.81065714175440506, 0.41913495986824945, 0.63452089516284882, 0.47264430310246486, 0.57147486659791213, 0.21991458204144373, 0.38834460899030065, 0.54972737682847062, 0.27390509643660554, 0.72169954128460811, 0.54111955155425506, 0.71657811017467521, 0.57796142296898889, 0.59591419125362932, 0.327344253731503, 0.49321650547059503, 0.54529178048174332, 0.48174206171033052, 0.63239374446544505, 0.37211770130746669, 0.85452235330554149, 0.86245046147222792, 0.40243454666000134, 0.57513767253291848, 0.7041108102696465, 0.76180893393434612, 0.67026356532774889]
'''
songs = ['AllTimeLow-StayAwake.mp3', 'Evanescence-BringMetoLife.mp3', 'BrittNicole-TheLostGetFound.mp3', 'CharityVance-IntoTheOcean.mp3', 'CD9-QuimicaEnComun.mp3', 'OneDirection-Moments.mp3', 'AlanWalker-Faded.mp3', 'BeaMiller-OpenYourEyes(DeepBlueSongspell).mp3', 'Oh,Come,AllYeFaithfulMusicVideoft.BYUVocalPointandBYUNoteworthy.mp3', 'AgainstTheCurrent-Wasteland.mp3', '5SecondsofSummer-Amnesia.mp3', 'ShowMetheMeaningofBeingLonely-(www.SongsLover.com).mp3', 'AllTimeLow-Backseatserenade.mp3', 'OneDirection-NaNaNa.mp3', 'CharityVance-HelloChange.mp3', 'BackstreetBoys-Aslongyouloveme.mp3', 'TaylorSwift-Red.mp3', 'AgainstTheCurrent-ForgetMeNow.mp3', 'ChristinaGrimmieandDiamondEyes-StayWithMe.mp3', 'ImagineDragons-Demons.mp3', 'EdSheeran-TheATeam.mp3', 'AdamLambert-BetterThanIKnowMyself.mp3', 'OneDirection-SameMistakes.mp3', 'OneDirection-Tellmealie.mp3', 'BackstreetBoys-Drowning.mp3', 'AllTimeLow-OldScarsFutureHearts.mp3', 'BackstreetBoys-IWantItThatWay.mp3', 'CD9-Termino(O.M.G.).mp3', 'OneDirection-Taken.mp3', 'DemiLovato-Nightingale.mp3', '5SecondsOfSummer-GreenLight.mp3', 'CelineDion-MyHeartWillGoOn.mp3']
valence = [0.8333333333333334, 0.16666666666666666, 0.16666666666666666, 0.25, 0.4166666666666667, 0.0, 0.08333333333333333, 0.5833333333333334, 0.5833333333333334, 0.0, 0.75, 0.9166666666666666, 0.3333333333333333, 0.9166666666666666, 0.0, 0.25, 0.16666666666666666, 0.4166666666666667, 0.25, 0.5, 0.0, 0.08333333333333333, 0.9166666666666666, 0.8333333333333334, 0.16666666666666666, 0.0, 0.75, 0.16666666666666666, 0.4166666666666667, 0.0, 0.4166666666666667, 0.5833333333333334]
arousal = [0.79133997999427474, 0.47857177108310645, 0.21981362134216345, 0.54132751457986927, 0.83781864386243943, 0.4682279975690572, 0.51417953922446291, 0.41561340423960264, 0.27522091799367188, 0.55986548448206508, 0.61233680028455084, 0.31863807255022819, 0.68701897605045703, 0.38831501963650927, 0.56703572359285881, 0.37927804259760012, 0.55744382136309101, 0.61314738031382432, 0.69889259500762724, 0.54032004887150165, 0.26818466525708462, 0.74481787730850424, 0.47916313052005832, 0.6535842173965194, 0.5317892058453223, 0.70832744011232518, 0.36382384480442498, 0.83311637194773802, 0.45897919739917287, 0.58397457698758348, 0.74758106591704809, 0.21886849312249831]


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
centroids={1:[0.75,0.75], 2:[0.25,0.75], 3:[0.25,0.25], 4:[0.75,0.25]}
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
print("Emotion 1")
for a in df['color']:
   if a=='r':
      print(songs[i])
   i=i+1
i=0
print ('')
print ('')
print("Emotion 2")
for a in df['color']:
   if a=='g':
      print(songs[i])
   i=i+1
i=0
print ('')
print ('')
print("Emotion 3")
for a in df['color']:
   if a=='b':
      print(songs[i])
   i=i+1
i=0
print ('')
print ('')
print("Emotion 4")
for a in df['color']:
   if a=='y':
      print(songs[i])
   i=i+1

#print(df['color'])
#print(df['color'][0])-----this is to get the individual colors for each song

