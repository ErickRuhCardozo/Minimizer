import os
import cv2
import pickle
import numpy as np
from PIL import Image
from pathlib import Path

classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
labels = {}
y_labels = []
x_train = []

for file in Path('people').rglob('*.jpg'):
		label = file.parent.name

		if not label in labels:
			labels[label] = current_id
			current_id += 1
		
		label_id = labels[label]		
		image = Image.open(str(file))
		image = image.convert('L') # grayscale
		image = image.resize((550, 550), Image.Resampling.LANCZOS)
		array = np.array(image, 'uint8')
		faces = classifier.detectMultiScale(array, scaleFactor=1.1, minNeighbors=3)

		for i, (x,y,w,h) in enumerate(faces):
			if i > 0:
				break

			roi = array[y:y+h, x:x+w]
			x_train.append(roi)
			y_labels.append(label_id)
			image = cv2.rectangle(array, (x, y), (x+w, y+h), (0, 0, 255), 2)


with open("labels.pkl", 'wb') as f:
	pickle.dump(labels, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trained.yml")