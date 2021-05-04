import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0 #assign
label_ids = {} #dictionary for lables
y_labels = [] # has the actual numbers that are related to the lables
x_train = [] # has the numbers of the actual pixel values

for root, dirs, files in os.walk(image_dir): # See all those images in image_dir
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("JPG"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower() #label for model
			print(label, path)

			# this code below describle how to create a dictionary with the lable or person's name
			# and the ID associated to it
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label] # an ID for that label

			print(label_ids)
			# verify this image, turn into a NUMPY arrray, GRAY
			pil_image = Image.open(path).convert("L") # grayscale
			size = (550, 550)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8") # convert it into NUMPY array
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5) # detect faces 

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi) # our training data
				y_labels.append(id_)



#print(np.array(y_labels))

#save label ids, writting byte
with open("pickles/face-labels.pickle", 'wb') as f: 
	pickle.dump(label_ids, f)
#train the OpenCV Recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")