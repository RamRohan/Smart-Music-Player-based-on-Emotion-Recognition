import numpy as np
import librosa
import subprocess
import glob
import os
import MySQLdb
import signal

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd=" ",  # your password
                     db="MUSIC")        # name of the data base


#list1 = subprocess.check_output(['ls'], cwd = '/home/radhakumaran/Documents/Projects/selfStudySemV/code/songs')
#print list1

#This first part finds all the files - you'll need to move all the music to a folder named 'songs' which is in the folder containing this code.
filepath = 'songs/*.mp3'
path = os.getcwd()
list2 = glob.glob(filepath)
for x in list2:
        r = x.replace(" ", "_")
        os.rename(x,r)
        #os.rename(os.path.join(path, x), os.path.join(path, x.replace(' ', '_')))

print list2

songs = []
valence = []
arousal = []
'''
cur = db.cursor()
cur.execute('DELETE FROM SONGS');
db.commit()
'''
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
        
        
        if(pos>4):
                pos = pos - 5
        else :
                pos = pos + 7
        
        
        val1 = pos/12.0

        valence = valence + [val1]
        
        arousal = arousal + [ar]
        songs = songs + [song[6:]]
        '''
        cur.execute('INSERT INTO SONGS VALUES (%s, %s, %s, %s, NULL)', (song[6:], song, val1, ar, ))
        db.commit()
        '''
        print song[6:], ar, val1


print 'songs :', songs
print 'valence :', valence
print 'arousal :', arousal

