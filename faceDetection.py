'''
import cv2
import sys

#cascPath = sys.argv[1]
#faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture('/home/radhakumaran/Videos/Joe Brooks - Superman.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

cap = cv2.VideoCapture('/home/radhakumaran/Videos/JoeBrooksSuperman.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/home/radhakumaran/Downloads/JoeBrooksSuperman.png',img)
    img = cv2.imread('/home/radhakumaran/Downloads/JoeBrooksSuperman.png')
    plt.imshow(img)
    plt.show()
    #cv2.imshow('frame',gray)
    if cv2.waitKey(100000) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()

