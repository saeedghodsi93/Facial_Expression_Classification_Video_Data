import os
import cv2
import numpy as np
import itertools
import matplotlib.pyplot as plt
import menpo.io as mio
from menpo.visualize import print_progress
from menpodetect import load_dlib_frontal_face_detector
from menpofit.aam import PatchAAM
from menpo.feature import fast_dsift
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional

# initialization
def init():
	
	print 'Initialization...'
	
	# input and output directories for reading and writing files
	inputdir = 'input files'
	outputdir = 'output files'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	return inputdir, outputdir

# load dataset images and landmarks
def load_dataset(inputdir):

	training_images = []
	datasetdir = inputdir + '/training dataset'
	for img in print_progress(mio.import_images(datasetdir, verbose=True)):
		# convert to greyscale
		if img.n_channels == 3:
			img = img.as_greyscale()
			
		# crop to landmarks bounding box with an extra padding
		img = img.crop_to_landmarks_proportion(0.25)
		
		# append img to training images
		training_images.append(img)
		
	return training_images

# load test data from file
def load_test_data(inputdir):

	# load test image
	test_images = []
	imagesdir = inputdir + '/test images'
	for img in print_progress(mio.import_images(imagesdir, verbose=True)):
		test_images.append(img)

	return test_images

# load train and test data
def load_train_and_test_data(inputdir):

	print 'Loading Train and Test Data...'
	
	training_images = load_dataset(inputdir)
	test_images = load_test_data(inputdir)
	
	return training_images, test_images

# train model
def aam_training(training_images):	

	patch_aam = PatchAAM(training_images, group='PTS', patch_shape=[(15, 15), (23, 23)], diagonal=150, scales=(0.5, 1.0), holistic_features=fast_dsift, max_shape_components=20, max_appearance_components=150, verbose=True)
	fitter = LucasKanadeAAMFitter(patch_aam, lk_algorithm_cls=WibergInverseCompositional, n_shape=[5, 20], n_appearance=[30, 150])

	return fitter
	
# convert landmarks to standard opencv keypoints
def landmarks_to_keypoints(landmarks):

	keypoint_size = 5.0
	keypoints = []
	keypoints.append(cv2.KeyPoint(landmarks[1],landmarks[0], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[3],landmarks[2], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[5],landmarks[4], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[7],landmarks[6], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[9],landmarks[8], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[11],landmarks[10], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[13],landmarks[12], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[15],landmarks[14], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[17],landmarks[16], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[19],landmarks[18], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[21],landmarks[20], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[23],landmarks[22], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[25],landmarks[24], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[27],landmarks[26], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[29],landmarks[28], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[31],landmarks[30], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[33],landmarks[32], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[35],landmarks[34], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[37],landmarks[36], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[39],landmarks[38], keypoint_size))
	keypoints.append(cv2.KeyPoint(landmarks[41],landmarks[40], keypoint_size))
	
	return keypoints

#test image with model
def aam_fitting(fitter, test_images):

	# Load detector
	detect = load_dlib_frontal_face_detector()
	
	i = 1
	for img in test_images:
	
		# convert to greyscale
		if img.n_channels == 3:
			gray = img.as_greyscale()
		
		# detect bounding box
		bboxes = detect(gray)
		
		if len(bboxes)>0:
			
			# initial bounding box
			initial_bbox = bboxes[0]

			# fit image
			result = fitter.fit_from_bb(gray, initial_bbox, max_iters=[15, 5])
			
			# convert image landmarks to keypoints
			kp = landmarks_to_keypoints(result.final_shape.as_vector())
		
			# save image
			imagesdir = outputdir + '/AAM_Method'
			img = img.pixels_with_channels_at_back(out_dtype=np.uint8)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			img = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
			cv2.imwrite(os.path.join(imagesdir, 'image'+str(i)+'.jpg'), img)
			cv2.waitKey(0)
			
			i = i + 1
		
	return
	
# program main
if __name__ == '__main__':

	inputdir, outputdir = init()
	
	training_images, test_images = load_train_and_test_data(inputdir)
	
	fitter = aam_training(training_images)
	
	aam_fitting(fitter, test_images)
	
	cv2.destroyAllWindows()
	