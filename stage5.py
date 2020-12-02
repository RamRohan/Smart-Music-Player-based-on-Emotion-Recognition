import cv2
import numpy as np
import argparse
import time
import glob
import os
import random
import Update_Model
import time
import MySQLdb
import subprocess
import signal
import os

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd=" ",  # your password
                     db="MUSIC") 

video_capture = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
fishface = cv2.face.createFisherFaceRecognizer()
#try:
#    fishface.load("trained_emoclassifier.xml")
#except:
#    print("no xml found. Using --update will create one.")

parser = argparse.ArgumentParser(description="Options for the emotion-based music player") #Create parser object
parser.add_argument("--update", help="Call to grab new images and update the model accordingly", action="store_true") #Add --update argument
args = parser.parse_args() #Store any given arguments in an object

facedict = {}
emotions = ["calm", "angry", "happy", "sad"]
data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    path = 'dataset/'+ emotion + '/*'
    files = glob.glob(path)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels

def crop_face(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
    global n
    if n>1:
        facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

def update_model(emotions):
    print("Model update mode active")
    check_folders(emotions)
    time.sleep(1)
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("collected images, looking good! Now updating model...")
    Update_Model.update(emotions)
    print("Done!")

def check_folders(emotions): #check if folder infrastructure is there, create if absent
    for x in emotions:
        path = 'dataset/' + x 
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

def save_face(emotion):
    print("\n\nplease look " + emotion + " when the timer expires and keep the expression stable until instructed otherwise.")
    time.sleep(1)
    for i in range(0,5):#Timer to give you time to read what emotion to express
        print(5-i)
        time.sleep(1)
    #while len(facedict.keys()) < 16: #Grab 15 images for each emotion
        detect_face()
    for x in facedict.keys(): #save contents of dictionary to files
        cv2.imwrite('dataset/' + emotion + '/' + str(len(glob.glob('dataset/'+emotion+'/*')))+'.jpg', facedict[x])
    facedict.clear() #clear dictionary so that the next emotion can be stored

def recognize_emotion():
    predictions = []
    confidence = []
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    fishface.train(training_data, np.asarray(training_labels))
    for x in facedict.keys():
        pred, conf = fishface.predict(facedict[x])
        cv2.imwrite("images/" + x +".jpg", facedict[x])
        predictions.append(pred)
        confidence.append(conf)
    print("I think you're %s" %emotions[max(set(predictions), key=predictions.count)])
    emotion= emotions[max(set(predictions), key=predictions.count)]
    print emotion

    cur=db.cursor()
    cur.execute("SELECT PATH FROM SONGS WHERE EMOTION = %s",(emotion,))
    song_path = cur.fetchall();


    print song_path

    for item in song_path:
         print item[0]
         doc = subprocess.Popen(["start", "WAIT", "/home/Documents/Projects/selfStudySemV/code/"+item[0]], shell=True)
         print 'started'
         doc.wait()
         doc.kill()

def grab_webcamframe():
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image

n=0
def detect_face():

    '''cap = cv2.VideoCapture(0)
    ret, img = cap.read() 
 
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
 
        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray) 
 
        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
 
    # Display an image in a window
    cv2.imshow('img',img)
    '''
    clahe_image = grab_webcamframe()
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1: 
        faceslice = crop_face(clahe_image, face)
        global n
        cv2.imwrite("test"+str(n)+".png", faceslice)
        global n
        n=n+1
        return faceslice
    else:
        print("no/multiple faces detected, passing over frame")

while True:
    time.sleep(.300);
    print "detecting face"
    detect_face()
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    if args.update: #If update flag is present, call update function
        update_model(emotions)
        break
    elif len(facedict) == 3: #otherwise it's regular a runtime, continue normally with emotion detection functionality
        recognize_emotion()
        break


