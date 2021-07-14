import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

IM_SIZE = 50

DATADIR = "D:\CARLA_0.9.5\PythonAPI\examples"
CATEGORIES = ['input', 'output']

X = []
y = []
count = 0
for category in CATEGORIES:
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		try:
			if category == 'input':
				count += 1
				img_array = cv2.imread(os.path.join(path, img))
				X.append(img_array)
			else:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				y.append(img_array)
		except Exception as e:
			pass

X = np.array(X).reshape(-1, 480, 360, 3)
y = np.array(y).reshape(-1, 480, 360, 1)

print(count)

import pickle

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()