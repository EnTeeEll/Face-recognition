import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

#load lables from face-labels.pickle and reverse it
labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()} 
	

cap = cv2.VideoCapture(0)

img = cv2.imread("8.jpg")
cv2.imshow('RGB', img)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) # detect all faces in frame

    for (x, y, w, h) in faces:
    	#print(x,y,w,h)
    	roi_gray = gray[y:y+h, x:x+w] 	#(ycord_start, ycord_end)
    	roi_color = frame[y:y+h, x:x+w]

    	# recognize? deep learned model predict keras tensorflow pytorch scikit learn
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=45: #and conf <= 85 :
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255) #BGR 0-255
    		stroke = 2 # how thick do you want the line actually be
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

    	img_item = "test.png"
    	cv2.imwrite(img_item, roi_color)

    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w #width
    	end_cord_y = y + h #height
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
