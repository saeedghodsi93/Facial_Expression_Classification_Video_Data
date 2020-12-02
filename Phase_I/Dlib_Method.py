import os
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
import menpo.io as mio
from menpo.visualize import print_progress
import dlib

# initialization
def init():
	
	print 'Initialization...'
	
	# input and output directories for reading and writing files
	inputdir = 'input files'
	outputdir = 'output files'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	face_cascade = cv2.CascadeClassifier(os.path.join(inputdir, 'haarcascades/haarcascade_frontalface_default.xml'))
	
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(os.path.join(inputdir, 'shape_predictor_68_face_landmarks.dat'))
	
	return inputdir, outputdir, face_cascade, detector, predictor

# load test data from file
def load_test_data(inputdir):

	# load test image
	test_images = []
	imagesdir = inputdir + '/test images'
	for img in print_progress(mio.import_images(imagesdir, verbose=True)):
		test_images.append(img)

	return test_images

#test image with model
def test(face_cascade, detector, predictor, test_images):

	i = 1
	for img in test_images:
	
		# convert to greyscale
		if img.n_channels == 3:
			gray = img.as_greyscale()
		img = img.pixels_with_channels_at_back(out_dtype=np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		gray = gray.pixels_with_channels_at_back(out_dtype=np.uint8)
		
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(5,5), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	
		dets = detector(gray, 1)
		for k, d in enumerate(dets):
			parts = predictor(gray, d).parts()
			landmarks = np.matrix([[p.x, p.y] for p in parts])
			for lm in landmarks:
				cv2.circle(img, (lm.tolist()[0][0],lm.tolist()[0][1]), 2, (0,255,255), -1)
				
		# save image
		imagesdir = outputdir + '/Dlib_Method'
		cv2.imwrite(os.path.join(imagesdir, 'image'+str(i)+'.jpg'), img)
		cv2.waitKey(0)
		
		i = i + 1
		
	return

# program main
if __name__ == '__main__':

	inputdir, outputdir, face_cascade, detector, predictor = init()
	
	test_images = load_test_data(inputdir)
	
	test(face_cascade, detector, predictor, test_images)
	
	cv2.destroyAllWindows()
	