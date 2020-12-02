import os
import cv2
import numpy as np
import random
import cPickle as pickle
import itertools
import matplotlib.pyplot as plt
import menpo.io as mio
from menpo.visualize import print_progress
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.externals import joblib
from scipy.stats import norm

# initialization
def init():
	
	print 'Initialization...'
	
	# input and output directories for reading and writing files
	inputdir = 'input files'
	outputdir = 'output files'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	# dataset directory
	datasetdir = inputdir + '/training dataset'
	
	# reload dataset or load from file
	retrain_classifier = False
	
	# diagonal of rescaled face images
	face_diagonal = 300
	
	# load face detector
	face_detector = cv2.CascadeClassifier(os.path.join(inputdir, 'haarcascades/haarcascade_frontalface_default.xml'))
		
	# initiate detectors and descriptors
	feature_point_detectors = []
	feature_point_descriptors = []
	feature_point_detectors_local = []
	fast = cv2.FeatureDetector_create('PyramidFAST')
	gftt = cv2.FeatureDetector_create('PyramidGFTT')
	mser = cv2.FeatureDetector_create('PyramidMSER')
	sift = cv2.SIFT()
	surf = cv2.SURF(500)
	brisk = cv2.BRISK(thresh=10)
	orb = cv2.ORB()
	brief = cv2.DescriptorExtractor_create('BRIEF')
	freak = cv2.DescriptorExtractor_create('FREAK')
	feature_point_detectors.append(fast)
	feature_point_detectors.append(surf)
	feature_point_detectors.append(mser)
	feature_point_detectors_local.append(mser)
	feature_point_descriptors.append(sift)
	feature_point_descriptors.append(brief)
	feature_point_descriptors.append(orb)
	feature_point_descriptors.append(brisk)
	feature_point_descriptors.append(freak)
	
	# number of labels, 12 or 19
	n_labels = 19
	max_labels = 21
	
	# prediction score thresholds
	high_threshold = 0.6
	low_threshold = 0.4
	ratio_threshold = 1.0
	
	return inputdir, outputdir, datasetdir, retrain_classifier, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, feature_point_detectors_local, n_labels, max_labels, high_threshold, low_threshold, ratio_threshold

# load dataset images and landmarks
def load_dataset(datasetdir, face_diagonal):

	training_images = []
	for img in print_progress(mio.import_images(datasetdir, verbose=True)):
		# convert to greyscale
		if img.n_channels == 3:
			img = img.as_greyscale()
			
		# crop to landmarks bounding box with an extra padding
		img = img.crop_to_landmarks_proportion(0.15)
		
		# rescale image diagonal to face_diagonal
		img = img.rescale(face_diagonal/img.diagonal())
		
		# append img to training images
		training_images.append(img)
		
	return training_images

# convert landmarks to standard opencv keypoints
def landmarks_to_keypoints(landmarks):

	landmarks = landmarks.get('PTS').lms.as_vector()
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
	
	labels = []
	labels.append(int(1))
	labels.append(int(2))
	labels.append(int(3))
	labels.append(int(4))
	labels.append(int(5))
	labels.append(int(6))
	labels.append(int(7))
	labels.append(int(8))
	labels.append(int(9))
	labels.append(int(10))
	labels.append(int(11))
	labels.append(int(12))
	labels.append(int(13))
	labels.append(int(14))
	labels.append(int(15))
	labels.append(int(16))
	labels.append(int(17))
	labels.append(int(18))
	labels.append(int(19))
	labels.append(int(20))
	labels.append(int(21))
	
	return keypoints, labels

# create some samples around each landmark
def create_positive_samples(img, keypoints, labels, feature_point_detectors_local):

	# mask image and keep just small area around estimated landmark location
	if img.n_channels == 3:
		img = img.as_greyscale()
		
	landmark_radius = 10
	temp_keypoints = []
	temp_labels = []
	for kp, lb in itertools.izip(keypoints, labels):
		
		img_masked = img.pixels_with_channels_at_back(out_dtype=np.uint8)
		mask_data = np.zeros((img_masked.shape[0],img_masked.shape[1]), np.uint8)
		cv2.circle(mask_data,(int(kp.pt[0]),int(kp.pt[1])),landmark_radius,255,-1)
		img_masked = cv2.bitwise_and(img_masked, img_masked, mask=mask_data)
		
		# find the keypoints
		loc_kp = []
		for feature_point_detector in feature_point_detectors_local:
			k = feature_point_detector.detect(img_masked,None)
			loc_kp.extend(k)
		
		# label keypoints in binary format
		for k in loc_kp:
			diff = np.array([kp.pt[0]-k.pt[0],kp.pt[1]-k.pt[1]])
			if np.linalg.norm(diff)<landmark_radius:
				temp_labels.append(lb)
				temp_keypoints.append(k)
				
	# show image keypoints
	#img2 = cv2.drawKeypoints(img.pixels_with_channels_at_back(out_dtype=np.uint8), temp_keypoints, None, color=(0,255,0), flags=0)
	#cv2.imshow('img',img2)
	#cv2.waitKey(0)
		
	keypoints = keypoints+temp_keypoints
	labels = labels+temp_labels
		
	return temp_keypoints, temp_labels
	
# sample some points with negative label from the image
def create_negative_samples(img, keypoints, labels):
	
	min_distance_from_landmark = 10
	n_samples = 5*max_labels
	keypoint_size = 5.0
	
	# convert image back to standard opencv format, and get image dimensions
	face = img.pixels_with_channels_at_back(out_dtype=np.uint8)
	rows = int(face.shape[0])
	cols = int(face.shape[1])
	
	# choose some sample points from whithin the face image
	counter = 0
	negative_points = []
	negative_labels = []
	while counter<n_samples:
		valid = True
		random_point_x = random.randint(1,cols)
		random_point_y = random.randint(1,rows)
		for kp in keypoints:
			diff = np.array([kp.pt[0]-random_point_x,kp.pt[1]-random_point_y])
			if np.linalg.norm(diff)<min_distance_from_landmark:
				valid = False
				
		if valid==True:
			negative_points.append(cv2.KeyPoint(random_point_x,random_point_y,keypoint_size))
			negative_labels.append(int(max_labels+1))
			counter = counter + 1
	
	keypoints = keypoints+negative_points
	labels = labels+negative_labels
	
	return keypoints, labels
	
# calculate landmarks location map	
def calculate_location_map(images):

	landmarks_location_map = []
	for img in images:
	
		# convert image landmarks to keypoints
		keypoints, labels = landmarks_to_keypoints(img.landmarks)
		
		if len(landmarks_location_map)==0:
			for kp in keypoints:
				pt_list = []
				pt_x = []
				pt_y = []
				pt_x.append(int(kp.pt[0]))
				pt_y.append(int(kp.pt[1]))
				pt_list.append(pt_x)
				pt_list.append(pt_y)
				landmarks_location_map.append(pt_list)
		else:
			for kp,lb in itertools.izip(keypoints,labels):
				landmarks_location_map[lb-1][0].append(int(kp.pt[0]))
				landmarks_location_map[lb-1][1].append(int(kp.pt[1]))
	
	return landmarks_location_map
 
# fit gaussian distribution to location map (for each landmark)
def fit_gaussian(landmarks_location_map):
 
	landmarks_location_gaussian = []
	for location in landmarks_location_map:
		gaussian = []
		mu_x, std_x = norm.fit(location[0])
		mu_y, std_y = norm.fit(location[1])
		gaussian.append(int(mu_x))
		gaussian.append(int(std_x))
		gaussian.append(int(mu_y))
		gaussian.append(int(std_y))
		landmarks_location_gaussian.append(gaussian)
		
	return landmarks_location_gaussian
	
# calculate training features
def calculate_training_features(images, feature_point_descriptors, feature_point_detectors_local):
	
	keypoints = []
	descriptors = []
	labels = []
	for feature_point_descriptor in feature_point_descriptors:
		kps = []
		dess = []
		lbs = []
		for img in images:
		
			# convert image landmarks to keypoints
			kp, lb = landmarks_to_keypoints(img.landmarks)
		
			# create samples for local classifiers
			kp, lb = create_positive_samples(img, kp, lb, feature_point_detectors_local)
			
			# sample some points with negative label from the image, and add to lists
			kp, lb = create_negative_samples(img, kp, lb)
			
			# convert image back to standard opencv format
			img = img.pixels_with_channels_at_back(out_dtype=np.uint8)
			#img = np.array(img.as_PILImage())
			
			# show image keypoints
			#img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
			#cv2.imshow('img',img2)
			#cv2.waitKey(0)
			
			# compute the descriptors
			kp, des = feature_point_descriptor.compute(img, kp)
			
			# append labels and descritors to the lists
			kps.append(kp)
			dess.append(des)
			lbs.append(lb)
			
		keypoints.append(kps)
		descriptors.append(dess)
		labels.append(lbs)
		
	return keypoints, descriptors, labels
	
# keep just desired landmarks and set others as negative class
def correct_labels(n_labels, max_labels, lb):
	
	if n_labels==12:
		if lb==2 or lb==5 or lb==8 or lb==11 or lb==13 or lb==14 or lb==16 or lb==17 or lb==19:
			return max_labels+1
		else:
			return lb
			
	elif n_labels==19:
		if lb==13 or lb==17:
			return max_labels+1
		else:
			return lb
		
	return max_labels+1
	
# load training data from file or reload
def load_training_data(datasetdir, face_diagonal, feature_point_descriptors, feature_point_detectors_local):
	
	print '\tLoading Training Data...'
	
	training_images = load_dataset(datasetdir, face_diagonal)
	
	landmarks_location_map = calculate_location_map(training_images)
	
	landmarks_location_gaussian = fit_gaussian(landmarks_location_map)
	
	keypoints, descriptors, labels = calculate_training_features(training_images, feature_point_descriptors, feature_point_detectors_local)
		
	return descriptors, labels, landmarks_location_gaussian

# create training feature vectors
def create_training_feature_vectors(training_descriptors, training_labels, n_labels, max_labels):
	
	print '\tCreating Training Feature Vectors...'
	
	temp_training_features = []
	temp_training_labels = []
	for descriptors,labels in itertools.izip(training_descriptors,training_labels):
		temp_features = []
		temp_labels = []
		for dess,lbs in itertools.izip(descriptors,labels):
			for des,lb in itertools.izip(dess,lbs):
				temp_features.append(des)
				temp_labels.append(correct_labels(n_labels, max_labels, lb))
		temp_training_features.append(temp_features)
		temp_training_labels.append(temp_labels)
	training_features = temp_training_features
	training_labels = temp_training_labels
	
	return training_features, training_labels

# show mean face landmarks on the test face image
def show_mean_face(face, landmarks_location_gaussian, win):

	# convert points to keypoints
	keypoint_size = 5.0
	kp = []
	offset_x = win[0]
	offset_y = win[1]
	for location in landmarks_location_gaussian:
		point_x = location[0]+offset_x
		point_y = location[2]+offset_y
		kp.append(cv2.KeyPoint(point_x,point_y, keypoint_size))
		
	# show feature points
	img = cv2.drawKeypoints(face, kp, None, color=(0,255,0), flags=0)
	cv2.imshow('img',img)
	cv2.waitKey(0)

	return
	
# feature point detection and description
def calculate_test_features(images, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, landmarks_location_gaussian):

	images_out = []
	images_face_windows = []
	keypoints = []
	descriptors = []
	for img in images:
	
		# convert image back to standard opencv format
		if img.n_channels == 3:
			gray = img.as_greyscale()
		face = gray.pixels_with_channels_at_back(out_dtype=np.uint8)
		
		# detect face in the image
		face_window = []
		faces = face_detector.detectMultiScale(face, scaleFactor=1.2, minNeighbors=5, minSize=(5,5), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		if len(faces)>0:
			(x,y,w,h) = faces[0]
			x1 = x
			x2 = x + w
			y1 = y + int(h*0.15)
			y2 = y + int(h*1.15)
			d = img.diagonal()
			x1_normalized = x1/d
			y1_normalized = y1/d
			x2_normalized = x2/d
			y2_normalized = y2/d

			# rescale image diagonal to face_diagonal
			img = img.rescale(face_diagonal/img.crop((y1,x1),(y2,x2)).diagonal())
			
			d = img.diagonal()
			x1 = int(x1_normalized*d)
			y1 = int(y1_normalized*d)
			x2 = int(x2_normalized*d)
			y2 = int(y2_normalized*d)
			face_window.append(x1)
			face_window.append(y1)
			face_window.append(x2)
			face_window.append(y2)
			
			# create a masked copy of the image, containing only face
			if img.n_channels == 3:
				gray = img.as_greyscale()
			face_masked = gray.pixels_with_channels_at_back(out_dtype=np.uint8)
			mask_data = np.zeros((face_masked.shape[0],face_masked.shape[1]), np.uint8)
			cv2.rectangle(mask_data,(x1,y1),(x2,y2),255,-1)
			face_masked = cv2.bitwise_and(face_masked, face_masked, mask=mask_data)
			
		else:
			face_window.append(int(1))
			face_window.append(int(1))
			face_window.append(face.shape[1])
			face_window.append(face.shape[0])
			face_masked = face
			
		# show face masked
		#cv2.imshow('img',face_masked)
		#cv2.waitKey(0)
	
		# find the keypoints
		temp_kp = []
		for feature_point_detector in feature_point_detectors:
			kp = feature_point_detector.detect(face_masked,None)
			temp_kp.extend(kp)
		kp = temp_kp
		
		# show feature points
		#img2 = cv2.drawKeypoints(face_masked, kp, None, color=(0,255,0), flags=0)
		#cv2.imshow('img',img2)
		#cv2.waitKey(0)

		# show mean face on the test face image
		#show_mean_face(face_masked, landmarks_location_gaussian, face_window)
		
		# compute the descriptors
		kps = []
		dess = []
		for feature_point_descriptor in feature_point_descriptors:
			kp, des = feature_point_descriptor.compute(face, kp)
			kps.append(kp)
			dess.append(des)
		
		# append img, keypoints and descritors to the lists
		images_out.append(img)
		images_face_windows.append(face_window)
		keypoints.append(kps)
		descriptors.append(dess)
	
	return images_out, images_face_windows, keypoints, descriptors
	
# load test data from file
def load_test_data(inputdir, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, landmarks_location_gaussian):
	
	print '\tLoading Test Data...'
	
	# load test image
	test_images = []
	imagesdir = inputdir + '/test images'
	for img in print_progress(mio.import_images(imagesdir, verbose=True)):
		test_images.append(img)

	test_images, test_images_face_windows, keypoints, descriptors = calculate_test_features(test_images, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, landmarks_location_gaussian)
	
	return test_images, test_images_face_windows, keypoints, descriptors
	
# create test feature vectors
def create_test_feature_vectors(test_descriptors):
	
	print '\tCreating Test Feature Vectors...'
	
	test_features = []
	for descriptors in test_descriptors:
		features = []
		for descriptor in descriptors:
			feature = []
			for des in descriptor:
				feature.append(des)
			features.append(feature)
		test_features.append(features)
		
	return test_features

# load train and test data
def load_train_and_test_data(inputdir, datasetdir, retrain_classifier, n_labels, max_labels, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, feature_point_detectors_local):

	print 'Loading Train and Test Data...'
		
	# predicted = cross_val_predict(classifier, training_features, training_labels, cv=5)
	# confusion_matrix = np.zeros((np.max(training_labels),np.max(training_labels)))
	# for lb,pr in itertools.izip(training_labels,predicted):
		# confusion_matrix[lb-1,pr-1] = confusion_matrix[lb-1,pr-1] + 1
	# np.savetxt(os.path.join(outputdir, 'Confusion Matrix.txt'), confusion_matrix, delimiter='\t', fmt='%d')
	# print metrics.accuracy_score(training_labels, predicted) 
	
	if retrain_classifier==True:
	
		# load training data
		training_descriptors, training_labels, landmarks_location_gaussian = load_training_data(datasetdir, face_diagonal, feature_point_descriptors, feature_point_detectors_local)
	
		# create training feature vectors
		training_features, training_labels = create_training_feature_vectors(training_descriptors, training_labels, n_labels, max_labels)
	
		# train the classifier
		print '\tTraining Classifiers...'
		classifiers = []
		for features,labels in itertools.izip(training_features,training_labels):
			#classifier = ExtraTreesClassifier(n_estimators=100, n_jobs=4)						# initiate classifier!
			classifier = MLPClassifier(solver='adam',alpha=1,random_state=1)
			classifier.fit(features,labels)
			classifiers.append(classifier)
	
		# save trained classifier, and locations map
		joblib.dump(classifiers, os.path.join(inputdir, 'classifiers.p'))
		pickle.dump(landmarks_location_gaussian, open(os.path.join(inputdir, 'landmarks_location_gaussian.p'), 'wb'))
		
	else:
		
		# load pre-trained classifier
		print '\tLoading Classifier...'
		classifiers = joblib.load(os.path.join(inputdir, 'classifiers.p'))
		landmarks_location_gaussian = pickle.load(open(os.path.join(inputdir, 'landmarks_location_gaussian.p'), 'rb'))
		
	# load test data
	test_images, test_images_face_windows, test_keypoints, test_descriptors = load_test_data(inputdir, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, landmarks_location_gaussian)
	
	# create test feature vectors
	test_features = create_test_feature_vectors(test_descriptors)

	return classifiers, landmarks_location_gaussian, test_features, test_images, test_images_face_windows, test_keypoints
	
# geometric probability of keypoint, belonging to label
def calculate_geometric_probability(type, radial_distance, landmarks_location_gaussian, win, kp, lb):

	if lb==max_labels+1:
		geometric_probability = 0
		
	else:
		point_x = kp.pt[0]
		point_y = kp.pt[1]
		offset_x = win[0]
		offset_y = win[1]
		mean_x = landmarks_location_gaussian[lb-1][0]+offset_x
		std_x = landmarks_location_gaussian[lb-1][1]
		mean_y = landmarks_location_gaussian[lb-1][2]+offset_y
		std_y = landmarks_location_gaussian[lb-1][3]
		
		# return probability of input point, belonging to normal distribution around mean point
		if type=='probabilistic':
			probability_x = norm(mean_x,std_x).pdf(point_x)/norm(mean_x,std_x).pdf(mean_x)
			probability_y = norm(mean_y,std_y).pdf(point_y)/norm(mean_y,std_y).pdf(mean_y)
			geometric_probability = probability_x * probability_y
			
		# return 1 if point is within circle around mean point, and 0 otherwise
		elif type=='deterministic':
			if point_x>=mean_x-radial_distance*std_x and point_x<=mean_x+radial_distance*std_x and point_y>=mean_y-radial_distance*std_y and point_y<=mean_y+radial_distance*std_y:
				geometric_probability = 1
			else:
				geometric_probability = 0
				
		else:
			geometric_probability = 0
			
	return geometric_probability
	
# remove wrong point from the classifier output and choose best prediction for each lancdmark
def threshold_predicted_points(landmarks_location_gaussian, win, keypoints, predicted_labels, predicted_probabilities, max_labels, first_threshold, second_threshold, ratio_threshold, geometric_type, radial_distance):

	# remove points with negative predicted label
	temp_keypoints = []
	temp_predicted_labels = []
	temp_predicted_probabilities = []
	for kp,lb,pr in itertools.izip(keypoints,predicted_labels,predicted_probabilities):
		if (lb<=max_labels):
			temp_keypoints.append(kp)
			temp_predicted_labels.append(lb)
			temp_predicted_probabilities.append(pr)
	keypoints = temp_keypoints
	predicted_labels = temp_predicted_labels
	predicted_probabilities = temp_predicted_probabilities
	
	# keep points with high best to second best prediction score, and remove others
	temp_keypoints = []
	temp_predicted_labels = []
	temp_predicted_probabilities = []
	for kp,lb,pr in itertools.izip(keypoints,predicted_labels,predicted_probabilities):
		sorted_idx = np.argsort(-pr)
		best_idx = sorted_idx[0]
		second_best_idx = sorted_idx[1]
		best_score = pr[best_idx]
		second_best_score = pr[second_best_idx]
		if (best_score>first_threshold and best_score>ratio_threshold*second_best_score):
			temp_keypoints.append(kp)
			temp_predicted_labels.append(lb)
			temp_predicted_probabilities.append(best_score)
	keypoints = temp_keypoints
	predicted_labels = temp_predicted_labels
	predicted_probabilities = temp_predicted_probabilities
	
	# keep points with high geometric prediction score, and remove others
	temp_keypoints = []
	temp_predicted_labels = []
	temp_predicted_probabilities = []
	for kp,lb,pr in itertools.izip(keypoints,predicted_labels,predicted_probabilities):
		pr = calculate_geometric_probability(geometric_type,radial_distance,landmarks_location_gaussian,win,kp,lb) * pr
		if (pr>second_threshold):
			temp_keypoints.append(kp)
			temp_predicted_labels.append(lb)
			temp_predicted_probabilities.append(pr)
	keypoints = temp_keypoints
	predicted_labels = temp_predicted_labels
	predicted_probabilities = temp_predicted_probabilities
	
	# just keep best found point for each landmark
	temp_keypoints = []
	temp_predicted_labels = []
	temp_predicted_probabilities = []
	for i in range(1,max_labels+1):
		idx = 0
		max_probability = 0
		max_idx = -1
		for kp,lb,pr in itertools.izip(keypoints,predicted_labels,predicted_probabilities):
			if lb==i and pr>max_probability:
				max_probability = pr
				max_idx = idx
			idx = idx + 1
		if max_idx>=0:
			temp_keypoints.append(keypoints[max_idx])
			temp_predicted_labels.append(predicted_labels[max_idx])
			temp_predicted_probabilities.append(predicted_probabilities[max_idx])
	keypoints = temp_keypoints
	predicted_labels = temp_predicted_labels
	predicted_probabilities = temp_predicted_probabilities
		
	return keypoints, predicted_labels, predicted_probabilities
	
# combine results of different descriptors, in decision level
def fuse_descriptors_results(keypoints, predicted_labels, predicted_probabilities):
	
	# just keep best found point for each landmark
	temp_keypoints = []
	temp_predicted_labels = []
	for i in range(1,max_labels+1):
		max_probability = 0
		for kps, predicted_lbs, predicted_prs in itertools.izip(keypoints, predicted_labels, predicted_probabilities):
			for kp, predicted_lb, predicted_pr in itertools.izip(kps, predicted_lbs, predicted_prs):
				if predicted_lb==i and predicted_pr>max_probability:
					point = kp
					max_probability = predicted_pr
		
		if max_probability>0:
			temp_keypoints.append(point)
			temp_predicted_labels.append(int(i))
			
	keypoints = temp_keypoints
	predicted_labels = temp_predicted_labels
	
	return keypoints, predicted_labels
	
# threshold found points and combine different discriptors to choose best point for each landmark
def find_landmarks(landmarks_location_gaussian, win, keypoints, predicted_labels, predicted_probabilities, max_labels, first_threshold, second_threshold, ratio_threshold, geometric_type, radial_distance):

	temp_keypoints = []
	temp_predicted_labels = []
	temp_predicted_probabilities = []
	for kps, predicted_lbs, predicted_prs in itertools.izip(keypoints, predicted_labels, predicted_probabilities):
		
		# remove unreliable points and choose best found point for each landmark, with high threshold and deterministic geometric distance consideration
		kps, predicted_lbs, predicted_prs = threshold_predicted_points(landmarks_location_gaussian, win, kps, predicted_lbs, predicted_prs, max_labels, first_threshold, second_threshold, ratio_threshold, geometric_type, radial_distance)
		
		temp_keypoints.append(kps)
		temp_predicted_labels.append(predicted_lbs)
		temp_predicted_probabilities.append(predicted_prs)
		
	# combine descriptors results and choose best found point for each landmark
	keypoints, predicted_labels = fuse_descriptors_results(temp_keypoints, temp_predicted_labels, temp_predicted_probabilities)

	return keypoints, predicted_labels
	
# align mean landmarks location with the test image
def alignment(img, landmarks_location_gaussian, win, keypoints, predicted_labels):

	# try to estimate face position in the screen relative to the mean face
	if len(keypoints)>0:
		magnitude = []
		angle = []
		for kp, lb in itertools.izip(keypoints, predicted_labels):
			offset_x = win[0]
			offset_y = win[1]
			delta_x = kp.pt[0]-(landmarks_location_gaussian[lb-1][0]+offset_x)
			delta_y = kp.pt[1]-(landmarks_location_gaussian[lb-1][2]+offset_y)
			mag, ang = cv2.cartToPolar(delta_x, delta_y)
			magnitude.append(mag[0])
			angle.append(ang[0])
			
		mag = np.median(np.array(magnitude))
		ang = np.median(np.array(angle))
		delta_x, delta_y = cv2.cartToPolar(mag, ang)
		delta_x = int(delta_x[0][0])
		delta_y = int(delta_y[0][0])
		
	else:
		delta_x = 0
		delta_y = 0
	
	# convert image back to standard opencv format
	if img.n_channels == 3:
		gray = img.as_greyscale()
	face = gray.pixels_with_channels_at_back(out_dtype=np.uint8)
		
	# show mean face on the test image face, befor alignment
	#show_mean_face(face, landmarks_location_gaussian)
		
	temp_landmarks_location_gaussian = []
	for location in landmarks_location_gaussian:
		temp_location = []
		temp_location.append(location[0]+delta_x)
		temp_location.append(location[1])
		temp_location.append(location[2]+delta_y)
		temp_location.append(location[3])
		temp_landmarks_location_gaussian.append(temp_location)
	landmarks_location_gaussian = temp_landmarks_location_gaussian
	
	# show mean face on the test image face, after alignment
	#show_mean_face(face, landmarks_location_gaussian)
		
	return landmarks_location_gaussian

# making results better by estimating remaining landmarks position from found landmarks
def refinment(landmarks_location_gaussian, win, keypoints, predicted_labels):

	keypoint_size = 5.0
	temp_test_keypoints = []
	temp_test_predicted_labels = []
	for i in range(1,max_labels+1):
		label = correct_labels(n_labels,max_labels,i)
		if label<=max_labels:
			found = False
			for kp, lb in itertools.izip(keypoints, predicted_labels):
				if lb==label:
					temp_test_keypoints.append(kp)
					temp_test_predicted_labels.append(lb)
					found = True
			if found==False:
				offset_x = win[0]
				offset_y = win[1]
				temp_test_keypoints.append(cv2.KeyPoint(landmarks_location_gaussian[label-1][0]+offset_x,landmarks_location_gaussian[label-1][2]+offset_y,keypoint_size))
				temp_test_predicted_labels.append(label)
	keypoints = temp_test_keypoints
	predicted_labels = temp_test_predicted_labels
	
	return keypoints, predicted_labels

# classification
def classification(classifiers, landmarks_location_gaussian, test_image, test_images_face_windows, test_features, test_keypoints, feature_point_detectors, feature_point_descriptors, max_labels, high_threshold, low_threshold, ratio_threshold):

	print 'Classification...'
	
	temp_test_keypoints = []
	temp_test_predicted_labels = []
	for img, win, features, keypoints in itertools.izip(test_image, test_images_face_windows, test_features, test_keypoints):
	
		temp_keypoints = []
		temp_predicted_labels = []
		temp_predicted_probabilities = []
		for frs, kps, classifier in itertools.izip(features, keypoints, classifiers):
		
			# predict labels of the keypoints
			predicted_lbs = classifier.predict(frs)
			predicted_prs = classifier.predict_proba(frs)
			
			# append descriptor result to the list
			temp_keypoints.append(kps)
			temp_predicted_labels.append(predicted_lbs)
			temp_predicted_probabilities.append(predicted_prs)
			
		# find landmarks with high threshold value to estimate face position in the screen
		#high_keypoints, high_predicted_labels = find_landmarks(landmarks_location_gaussian, win, temp_keypoints, temp_predicted_labels, temp_predicted_probabilities, max_labels, high_threshold, high_threshold, ratio_threshold, 'deterministic', 3)
		# align mean landmarks locations with the test image face
		#temp_landmarks_location_gaussian = alignment(img, landmarks_location_gaussian, win, high_keypoints, high_predicted_labels)
		
		# find landmarks with high threshold value to estimate face position in the screen
		#high_keypoints, high_predicted_labels = find_landmarks(temp_landmarks_location_gaussian, win, temp_keypoints, temp_predicted_labels, temp_predicted_probabilities, max_labels, high_threshold, high_threshold, ratio_threshold, 'deterministic', 2)
		# align mean landmarks locations with the test image face
		#temp_landmarks_location_gaussian = alignment(img, temp_landmarks_location_gaussian, win, high_keypoints, high_predicted_labels)
		
		# find landmarks with low threshold value to to find landmarks
		keypoints, predicted_labels = find_landmarks(landmarks_location_gaussian, win, temp_keypoints, temp_predicted_labels, temp_predicted_probabilities, max_labels, high_threshold, low_threshold, ratio_threshold, 'deterministic', 1)
		
		# make results better
		keypoints, predicted_labels = refinment(landmarks_location_gaussian, win, keypoints, predicted_labels)
		
		temp_test_keypoints.append(keypoints)
		temp_test_predicted_labels.append(predicted_labels)
	
	test_keypoints = temp_test_keypoints
	test_predicted_labels = temp_test_predicted_labels
	
	return test_keypoints, test_predicted_labels
	
# save classification result on image
def save_result(images, keypoints, labels):

	print 'Saving Result...'
	
	i = 1
	for img, kps, lbs in itertools.izip(images, keypoints, labels):
	
		# convert image back to standard opencv format
		img = img.pixels_with_channels_at_back(out_dtype=np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		
		# show predicted label on image
		font = cv2.FONT_HERSHEY_SIMPLEX
		for kp,lb in itertools.izip(kps,lbs):
			cv2.circle(img,(int(kp.pt[0]),int(kp.pt[1])), 2, (0,255, -1))
			cv2.putText(img, str(lb), (int(kp.pt[0]),int(kp.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0,0,255))
		
		# resize frame image by factor 1.5 to feet the screen, and show it
		img = cv2.resize(img, None, fx=1.5, fy=1.5)
		
		# save output image file
		imagesdir = outputdir + '/Main_Method'
		cv2.imwrite(os.path.join(imagesdir, 'image'+str(i)+'.jpg'), img)
		#cv2.imshow('img',img)
		#cv2.waitKey(0)
		i = i + 1
		
	return
	
# program main
if __name__ == '__main__':

	inputdir, outputdir, datasetdir, retrain_classifier, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, feature_point_detectors_local, n_labels, max_labels, high_threshold, low_threshold, ratio_threshold = init()
	
	classifiers, landmarks_location_gaussian, test_features, test_image, test_images_face_windows, test_keypoints = load_train_and_test_data(inputdir, datasetdir, retrain_classifier, n_labels, max_labels, face_detector, face_diagonal, feature_point_detectors, feature_point_descriptors, feature_point_detectors_local)
	
	test_keypoints, test_labels = classification(classifiers, landmarks_location_gaussian, test_image, test_images_face_windows, test_features, test_keypoints, feature_point_detectors, feature_point_descriptors, max_labels, high_threshold, low_threshold, ratio_threshold)
	
	save_result(test_image, test_keypoints, test_labels)
	
	cv2.destroyAllWindows()
	