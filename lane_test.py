import numpy as np
import cv2
from scipy.misc import imresize
from IPython.display import HTML
from keras.models import load_model


# Class to average lanes with
class Lanes():
	def __init__(self):
		self.recent_fit = []
		self.avg_fit = []

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color    
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def road_lines(image):
	""" Takes in a road image, re-sizes for the model,
	predicts the lane to be drawn from the model in G color,
	recreates an RGB image of a lane and merges with the
	original road image.
	"""

	# Get image ready for feeding into model
	image_rezise = imresize(image, (360, 480, 3))
	small_img = imresize(image, (480, 360, 3))
	small_img = np.array(small_img)
	small_img = small_img[None,:,:,:]

	# Make prediction with neural network (un-normalize value by multiplying by 255)
	prediction = model.predict(small_img)[0] * 255

	# Add lane prediction to list for averaging
	lanes.recent_fit.append(prediction)
	# Only using last five for average
	if len(lanes.recent_fit) > 5:
		lanes.recent_fit = lanes.recent_fit[1:]

	# Calculate average detection
	lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

	# Generate fake R & B color dimensions, stack with G
	blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
	lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

	# Re-size to match the original image
	lane_image = imresize(lane_drawn, (360, 480, 3))
	imshape = lane_image.shape
	# lower_right = [imshape[1]-imshape[1]/9,imshape[0]]
	# top_left = [imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10]
	# top_right = [imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10]
	# vertices = [np.array([[0,360], [0,240], [120,180], [360,180], [480, 240], [480,360]],dtype=np.int32)]
	# roi_image = region_of_interest(lane_image, vertices)
	roi_image = lane_image
	# print(small_img.shape, lane_image.shape)
	# Merge the lane drawing onto the original image
	result = cv2.addWeighted(image_rezise, 1, roi_image, 1, 0)

	return result, roi_image


if __name__ == '__main__':
	# Load Keras model
	model = load_model('full_CNN_model3.h5')
	# Create lanes object
	lanes = Lanes()

	# Where to save the output video

	image = cv2.imread('./input/30.png')
	result, lane_image = road_lines(image)
	cv2.imshow('result', result)
	cv2.waitKey(0)
	cv2.imshow('lane_image', lane_image)
	cv2.waitKey(0)