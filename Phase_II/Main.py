import os
import cv2
import numpy as np
import random
import math
import cPickle as pickle
import itertools
import matplotlib.pyplot as plt
import dlib
from skimage.feature import local_binary_pattern
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from collections import Counter

# initialization
def init():
	
	print 'Initialization...'
	
	# input and output directories for reading and writing files
	inputdir = 'input files'
	outputdir = 'output files'
	if not os.path.exists(outputdir):
		os.mkdir(outputdir)

	# test method, 'cross' for cross validation, 'images' for videos as images format, and 'videos' for videos in standard format
	test_method = 'cross'
	
	# cross validation method, '2fold', '4fold', '10fold', 'loso'
	cross_validation_method = '2fold'
	
	# dataset directory
	datasetdir = os.path.join(inputdir, 'dataset')
	
	# reload dataset or load from file
	reload_dataset = False
	
	# diagonal of rescaled face images
	region_width = 200.0

	# LPB parameters
	radius = 1
	n_points = 8 * radius

	# load face detector
	face_detector = dlib.get_frontal_face_detector()
	landmark_detector = dlib.shape_predictor(os.path.join(inputdir, 'shape_predictor_68_face_landmarks.dat'))
	
	# classifier
	#classifier = ExtraTreesClassifier(n_estimators=300, n_jobs=4)
	classifier = Pipeline([('feature_selection', SelectFromModel(LinearSVC(loss='squared_hinge', penalty='l1', dual=False))), ('classification', ExtraTreesClassifier(n_estimators=300, n_jobs=4))])
	
	# VideoWriter parameters
	fourcc = cv2.cv.CV_FOURCC(*'DIVX')
	frame_rate = 4.0
	resolution = (640,480)
	
	return inputdir, outputdir, datasetdir, reload_dataset, face_detector, landmark_detector, region_width, n_points, radius, cross_validation_method, classifier, fourcc, frame_rate, resolution, test_method

# load dataset images and landmarks
def load_dataset(datasetdir, face_detector, landmark_detector):

	print '\tLoading Dataset...'
	
	# dataset directories
	emotiondir = datasetdir + 'Emotion'
	imagesdir = datasetdir + 'cohn-kanade-images'
	landmarksdir = datasetdir + 'Landmarks'
	
	# subjects in emotion directory
	dataset = []
	subjects = [name for name in os.listdir(emotiondir) if os.path.isdir(os.path.join(emotiondir, name))]
	for subject in subjects:
		
		# expressions in subject directory
		subject_data = []
		emotion_subjectdir = emotiondir + '/' + subject
		expressions = [name for name in os.listdir(emotion_subjectdir) if os.path.isdir(os.path.join(emotion_subjectdir, name))]
		for expression in expressions:
		
			# file in expression directory
			expression_data = []
			emotion_subject_expressiondir = emotion_subjectdir + '/' + expression
			exp = [name for name in os.listdir(emotion_subject_expressiondir) if os.path.isfile(os.path.join(emotion_subject_expressiondir, name))]
			
			# check if file exists
			if exp!=[]:
				# append expression label to expression data list
				emotionfile = emotion_subject_expressiondir + '/' + exp[0]
				f = open(emotionfile)
				label_data = int(np.double(f.read()))
				f.close
				
				# corresponding images directory
				landmarks_data = []
				images_data = []
				images_subject_expressiondir = imagesdir + '/' + subject + '/' + expression
				imagesname = [name for name in os.listdir(images_subject_expressiondir) if os.path.isfile(os.path.join(images_subject_expressiondir, name))]
				temp_imagesname = []
				for imagename in imagesname:
					if imagename.find('.png')>0:
						temp_imagesname.append(imagename)
				imagesname = temp_imagesname
				temp_imagesname = []
				temp_imagesname.append(imagesname[0])
				temp_imagesname.append(imagesname[-1])
				imagesname = temp_imagesname
				for imagename in imagesname:
					imagefile = images_subject_expressiondir + '/' + imagename
					img = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
					images_data.append(img)
					# dets = face_detector(img, 1)
					# if len(dets)>0:
						# d = dets[0]
						# parts = landmark_detector(img, d).parts()
						# lms_data = []
						# landmarks = np.matrix([[p.x, p.y] for p in parts])
						# for lm in landmarks:
							# lm_data = []
							# lm_data.append(int(lm.tolist()[0][0]))
							# lm_data.append(int(lm.tolist()[0][1]))
							# lms_data.append(lm_data)
						# landmarks_data.append(lms_data)
					# else:
						# print 'failed to detect landmarks!'
					
				# correspoonding landmarks directory
				landmarks_subject_expressiondir = landmarksdir + '/' + subject + '/' + expression
				landmarksname = [name for name in os.listdir(landmarks_subject_expressiondir) if os.path.isfile(os.path.join(landmarks_subject_expressiondir, name))]
				temp_landmarksname = []
				for landmarkname in landmarksname:
					if landmarkname.find('.txt')>0:
						temp_landmarksname.append(landmarkname)
				landmarkname = temp_landmarksname
				temp_landmarksname = []
				temp_landmarksname.append(landmarksname[0])
				temp_landmarksname.append(landmarksname[-1])
				landmarksname = temp_landmarksname
				for landmarkname in landmarksname:
					landmarkfile = landmarks_subject_expressiondir + '/' + landmarkname
					lms_data = []
					f = open(landmarkfile, 'r')
					lms = f.readlines()
					f.close
					for lm in lms:
						if lm!='\n':
							lm_data = []
							lm_data.append(int(np.double(lm[3:16])))
							lm_data.append(int(np.double(lm[19:32])))
							lms_data.append(lm_data)
					landmarks_data.append(lms_data)
						
				# append label, landmarks, and images to the list
				expression_data.append(landmarks_data)
				expression_data.append(images_data)
				expression_data.append(label_data)
				
				# append expression data to subject data list
				subject_data.append(expression_data)
			
		# append sunject data to the dataset list
		dataset.append(subject_data)
	
	return dataset
	
# show landmarks location change
def show_landmarks(inputdir, first_frame, last_frame):

	offset = 200
	blank = cv2.imread(os.path.join(inputdir, 'blank.jpg'))
	for lm in first_frame:
		cv2.circle(blank,(int(lm[0]+offset),int(lm[1]+offset)), 5, (0,0,255), -1)
	for lm in last_frame:
		cv2.circle(blank,(int(lm[0]+offset),int(lm[1]+offset)), 5, (255,0,0), -1)
		
	cv2.imshow('img',blank)
	cv2.waitKey(0)
		
	return
	
# transform, rotate and scale face
def normalize_face(frame):
	
	mean_head_height = 250.0
	
	# create geometric features
	offset = frame[30]
	rotation_angle = np.arctan2(frame[45][1]-frame[36][1],frame[45][0]-frame[36][0])
	head_height = (frame[15][0]+frame[16][0])/2 - (frame[0][0]+frame[1][0])/2
			
	temp_frame = []
	for lm in frame:
		point = []
		point.append(int(lm[0]-offset[0]))
		point.append(int(lm[1]-offset[1]))
		temp_frame.append(point)
	frame = temp_frame
	
	temp_frame = []
	for lm in frame:
		rotated_point = []
		ang = -rotation_angle
		x1 = lm[0]
		y1 = lm[1]
		x2 = (x1*np.cos(ang))-(y1*np.sin(ang))
		y2 = (x1*np.sin(ang))+(y1*np.cos(ang))
		rotated_point.append(int(x2))
		rotated_point.append(int(y2))
		temp_frame.append(rotated_point)
	frame = temp_frame
	
	scale_factor = mean_head_height/head_height
	temp_frame = []
	for lm in frame:
		point = []
		point.append(int(lm[0]*scale_factor))
		point.append(int(lm[1]*scale_factor))
		temp_frame.append(point)
	frame = temp_frame
				
	return frame
	
# create LBP features for image
def create_appearance_features(img, lms, n_points, radius):

	horizontal_offset = 10
	vertical_offset = 10
	
	x1 = lms[17][0]-horizontal_offset
	x2 = lms[27][0]-horizontal_offset
	y1 = lms[19][1]-vertical_offset
	y2 = lms[41][1]+2*vertical_offset
	w = x2-x1
	lefteye = cv2.resize(img[y1:y2,x1:x2], None, fx=region_width/w, fy=region_width/w)
	pyramidlefteye00 = lefteye[:int(lefteye.shape[0]/2),:int(lefteye.shape[1]/2)]
	pyramidlefteye01 = lefteye[int(lefteye.shape[0]/2):,:int(lefteye.shape[1]/2)]
	pyramidlefteye10 = lefteye[:int(lefteye.shape[0]/2),int(lefteye.shape[1]/2):]
	pyramidlefteye11 = lefteye[int(lefteye.shape[0]/2):,int(lefteye.shape[1]/2):]
	x1 = lms[27][0]+horizontal_offset
	x2 = lms[26][0]+horizontal_offset
	y1 = lms[24][1]-vertical_offset
	y2 = lms[46][1]+2*vertical_offset
	w = x2-x1
	righteye = cv2.resize(img[y1:y2,x1:x2], None, fx=region_width/w, fy=region_width/w)
	pyramidrighteye00 = righteye[:int(righteye.shape[0]/2),:int(righteye.shape[1]/2)]
	pyramidrighteye01 = righteye[int(righteye.shape[0]/2):,:int(righteye.shape[1]/2)]
	pyramidrighteye10 = righteye[:int(righteye.shape[0]/2),int(righteye.shape[1]/2):]
	pyramidrighteye11 = righteye[int(righteye.shape[0]/2):,int(righteye.shape[1]/2):]
	x1 = lms[60][0]-2*horizontal_offset
	x2 = lms[54][0]+horizontal_offset
	y1 = (lms[50][1]+lms[52][1])/2-vertical_offset
	y2 = (lms[56][1]+lms[57][1]+lms[58][1])/3+vertical_offset
	w = x2-x1
	mouth = cv2.resize(img[y1:y2,x1:x2], None, fx=region_width/w, fy=region_width/w)					
	pyramidmouth00 = mouth[:int(mouth.shape[0]/2),:int(mouth.shape[1]/2)]
	pyramidmouth01 = mouth[int(mouth.shape[0]/2):,:int(mouth.shape[1]/2)]
	pyramidmouth10 = mouth[:int(mouth.shape[0]/2),int(mouth.shape[1]/2):]
	pyramidmouth11 = mouth[int(mouth.shape[0]/2):,int(mouth.shape[1]/2):]
	#cv2.imshow('img',lefteye)
	#cv2.waitKey(0)
	#cv2.imshow('img',righteye)
	#cv2.waitKey(0)
	#cv2.imshow('img',mouth)
	#cv2.waitKey(0)
	
	feature_vector = []
	lbp = local_binary_pattern(lefteye, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidlefteye00, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidlefteye01, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidlefteye10, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidlefteye11, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	
	lbp = local_binary_pattern(righteye, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidrighteye00, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidrighteye01, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidrighteye10, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidrighteye11, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	
	lbp = local_binary_pattern(mouth, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidmouth00, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidmouth01, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidmouth10, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
	lbp = local_binary_pattern(pyramidmouth11, n_points, radius, 'nri_uniform')
	x = np.histogram(lbp.ravel(), normed=True, bins=59, range=(0, 59))
	sum = np.double(np.sum(x[0]))
	hist = x[0]
	feature_vector.extend(hist)
 			
	return feature_vector
	
# align and normalize faces, and calculate features
def create_features(inputdir, dataset):
	
	print '\tCreating Feature Vectors...'
	
	dataset_data = []
	normalized_landmarks = []
	labels = []
	for subject_data in dataset:
	
		subject_landmarks = []
		subject_labels = []
		for expression_data in subject_data:
			landmarks_data = expression_data[0]
			images_data = expression_data[1]
			label_data = expression_data[2]
			
			if len(landmarks_data)!=2:
				print 'Error in Number of Frames!'
			
			expression_features = []
			expression_labels = []
		
			first_frame = landmarks_data[0]
			last_frame = landmarks_data[-1]
			
			# show landmarks, befor normalization
			#show_landmarks(inputdir,first_frame,last_frame)
	
			first_frame = normalize_face(first_frame)
			last_frame = normalize_face(last_frame)
			
			# show landmarks, after normalization
			#show_landmarks(inputdir,first_frame,last_frame)
			
			# neutral label
			feature_vector = []
			for flm,flm in itertools.izip(first_frame,first_frame):
				feature_vector.append(flm[0]-flm[0])
				feature_vector.append(flm[1]-flm[1])
			feature_vector.extend(create_appearance_features(images_data[0],landmarks_data[0],n_points,radius))
			expression_features.append(feature_vector)
			expression_labels.append(int(0))
			
			# expression label
			feature_vector = []
			for flm,llm in itertools.izip(first_frame,last_frame):
				feature_vector.append(llm[0]-flm[0])
				feature_vector.append(llm[1]-flm[1])
			feature_vector.extend(create_appearance_features(images_data[-1],landmarks_data[-1],n_points,radius))
			expression_features.append(feature_vector)
			expression_labels.append(label_data)
			
			subject_landmarks.append(expression_features)
			subject_labels.append(expression_labels)
			
		normalized_landmarks.append(subject_landmarks)
		labels.append(subject_labels)
	
	dataset_data.append(normalized_landmarks)
	dataset_data.append(labels)
	
	return dataset_data

# load training feature vectors
def load_dataset_features(inputdir, datasetdir, reload_dataset, face_detector, landmark_detector, region_width, n_points, radius):

	print 'Loading Dataset Data...'
	
	if reload_dataset==True:
		
		dataset = load_dataset(datasetdir, face_detector, landmark_detector)
		
		dataset_data = create_features(inputdir,dataset)
		
		pickle.dump(dataset_data, open(os.path.join(inputdir, 'dataset_data.p'), 'wb'))
		
	else:
	
		dataset_data = pickle.load(open(os.path.join(inputdir, 'dataset_data.p'), 'rb'))
	
	return dataset_data

# partition dataset to training and testing sets
def partition(dataset_data, method, indices, pivot):
	
	print '\tPartitioning Dataset...'
	
	n_subjects = len(dataset_data[0])
	if method=='2fold' or method=='4fold' or method=='10fold':
		k = int(method[:-4])
		if pivot<k-1:
			test_indices = indices[int(pivot*math.ceil(n_subjects/k)):int((pivot+1)*math.ceil(n_subjects/k))-1]
			training_indices = np.copy(indices)
			training_indices = np.delete(training_indices, range(int(pivot*math.ceil(n_subjects/k)),int((pivot+1)*math.ceil(n_subjects/k))))
		else:
			test_indices = indices[int(pivot*math.ceil(n_subjects/k)):]
			training_indices = indices[0:int(pivot*math.ceil(n_subjects/k))-1]
			
	elif method=='loso':
		test_indices = []
		test_indices.append(pivot)
		training_indices = indices[:]
		training_indices.remove(pivot)
		
	#print training_indices, test_indices
	
	training_data = []
	training_features = []
	training_labels = []
	test_data = []
	test_features = []
	test_labels = []
	[training_features.append(dataset_data[0][i]) for i in training_indices]
	[training_labels.append(dataset_data[1][i]) for i in training_indices]
	training_data.append(training_features)
	training_data.append(training_labels)
	[test_features.append(dataset_data[0][i]) for i in test_indices]
	[test_labels.append(dataset_data[1][i]) for i in test_indices]
	test_data.append(test_features)
	test_data.append(test_labels)
	
	return training_data, test_data
	
# create train and test features
def create_feature_vectors(data,phase):
	
	temp_features = []
	temp_labels = []
	features = data[0]
	labels = data[1]
	for subject_features,subject_labels in itertools.izip(features,labels):
		for expression_features,expression_label in itertools.izip(subject_features,subject_labels):
			if phase=='train':
				for fr, lb in itertools.izip(expression_features,expression_label):
					temp_features.append(fr)
					temp_labels.append(lb)
			elif phase=='test':
				fr = expression_features[-1]
				lb = expression_label[-1]
				temp_features.append(fr)
				temp_labels.append(lb)
	features = temp_features
	labels = temp_labels
	
	return features, labels

# feature selection
def feature_selection(training_features, training_labels, test_features):

	# perform anova test on training data, and choose most discriminating features
	n_features = 136
	anova = f_classif(training_features, training_labels)
	anova = anova[0]
	idx = np.argsort(anova)[::-1][:n_features]
	values = []
	[values.append(anova[i]) for i in idx]
	
	temp_training_features = []
	for feature in training_features:
		temp_feature = []
		[temp_feature.append(feature[i]) for i in idx]
		temp_training_features.append(temp_feature)
	training_features = temp_training_features
	
	temp_test_features = []
	for feature in test_features:
		temp_feature = []
		[temp_feature.append(feature[i]) for i in idx]
		temp_test_features.append(temp_feature)
	test_features = temp_test_features
	
	return training_features, test_features

# classification
def classification(classifier, training_features, training_labels, test_features):
	
	print '\tClassification...'
	
	classifier.fit(training_features,training_labels)
	
	predicted_labels = classifier.predict(test_features)
		
	return predicted_labels
	
def plot_confusion_matrix(cm):

	normalize=True
	classes = ['Neutral', 'Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
	
	cm = cm.astype('int')
	print('Confusion Matrix=')
	print(cm)
	
	norm_cm = np.copy(cm)
	norm_cm = norm_cm.astype('float')
	for row in range(0,cm.shape[0]):
		s = cm[row,:].sum()
		if s > 0:
			for col in range(0,cm.shape[0]):
				norm_cm[row,col] = np.double(cm[row,col]) / s
	print("Normalized Confusion Matrix=")
	print(norm_cm)
	
	# plot confusion matrix
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, int(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
	plt.savefig(os.path.join(outputdir, 'Confusion Matrix.jpg'))
	plt.show()
	
	# plot normalized confusion matrix
	plt.figure()
	plt.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Confusion matrix')
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = norm_cm.max() / 2.
	for i, j in itertools.product(range(norm_cm.shape[0]), range(norm_cm.shape[1])):
		plt.text(j, i, float("{0:.2f}".format(norm_cm[i, j])), horizontalalignment="center", color="white" if norm_cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.savefig(os.path.join(outputdir, 'Normalized Confusion Matrix.jpg'))
	plt.show()
	
	return
	
# cross validation test on dataset
def cross_validation(dataset_data, method, classifier):
	print 'Cross Validation...'
	
	n_tests = 10
	total_test_labels = []
	total_predicted_labels = []
	n_subjects = len(dataset_data[0])
	accuracies = []
	if method=='2fold' or method=='4fold' or method=='10fold':
	
		for i in range(0,n_tests):
		
			print '\tTest Number = ' + str(i)
			
			test_accuracies = []
			indices = np.random.permutation(n_subjects)
			k = int(method[:-4])
			for j in range(0,k):
			
				training_data, test_data = partition(dataset_data, method, indices, j)
				
				training_features, training_labels = create_feature_vectors(training_data,'train')
				
				test_features, test_labels = create_feature_vectors(test_data,'test')
				
				#training_features, test_features = feature_selection(training_features, training_labels, test_features)
				
				predicted_labels = classification(classifier, training_features, training_labels, test_features)
				
				total_test_labels.extend(test_labels)
				total_predicted_labels.extend(predicted_labels)
				test_accuracies.append(metrics.accuracy_score(test_labels, predicted_labels))
				
			accuracies.append(np.mean(test_accuracies))
			
	elif method=='loso':
			
		for i in range(0,len(dataset_data[0])):
		
			print '\tTest Number = ' + str(i)
			
			training_data, test_data = partition(dataset_data, method, range(0,n_subjects), i)
		
			training_features, training_labels = create_feature_vectors(training_data,'train')
			
			test_features, test_labels = create_feature_vectors(test_data,'test')
			
			#training_features, test_features = feature_selection(training_features, training_labels, test_features)
			
			if len(test_labels)>0:
			
				predicted_labels = classification(classifier, training_features, training_labels, test_features)
			
				total_test_labels.extend(test_labels)
				total_predicted_labels.extend(predicted_labels)
				accuracies.append(metrics.accuracy_score(test_labels, predicted_labels))
	
	print 'Accuracy= ' + str(np.mean(accuracies))
	
	# calculate total confusion matrix for all tests
	total_test_labels.append(0)
	total_predicted_labels.append(0)
	cnf_matrix = confusion_matrix(total_test_labels, total_predicted_labels)
	np.set_printoptions(precision=2)

	# plot confusion matrix
	plot_confusion_matrix(cnf_matrix)
	
	# save confusion matrix as text
	#np.savetxt(os.path.join(outputdir, 'Confusion Matrix.txt'), confusion_matrix, delimiter='\t', fmt='%d')
			
	return
	
# load test data from input folder, and create feature vectors
def load_test_data(inputdir, face_detector, landmark_detector, test_method):

	print 'Loading Test Data...'
	
	if test_method=='images':
		# test directory
		test_data = []
		testdir = inputdir + '/images'
		tests = [name for name in os.listdir(testdir) if os.path.isdir(os.path.join(testdir, name))]
		for test in tests:
		
			# load images and detect kandmarks
			expression_data = []
			landmarks_data = []		
			images_data = []
			expressiondir = testdir + '/' + test
			imagesname = [name for name in os.listdir(expressiondir) if os.path.isfile(os.path.join(expressiondir, name))]
			temp_imagesname = []
			for imagename in imagesname:
				if imagename.find('.png')>0:
					temp_imagesname.append(imagename)
			imagesname = temp_imagesname
			for imagename in imagesname:
				imagefile = expressiondir + '/' + imagename
				img = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
				images_data.append(img)
				dets = face_detector(img, 1)
				if len(dets)>0:
					d = dets[0]
					parts = landmark_detector(img, d).parts()
					lms_data = []
					landmarks = np.matrix([[p.x, p.y] for p in parts])
					for lm in landmarks:
						lm_data = []
						lm_data.append(int(lm.tolist()[0][0]))
						lm_data.append(int(lm.tolist()[0][1]))
						lms_data.append(lm_data)
					landmarks_data.append(lms_data)
				else:
					print 'failed to detect landmarks!'
			
			# first frame of video
			first_frame = landmarks_data[0]
			
			# normalize first frame
			first_frame = normalize_face(first_frame)
				
			# next frames of video
			feature_vectors = []
			for img, frame in itertools.izip(images_data, landmarks_data):
			
				# normalize current frame
				offset = frame[30]
				current_frame = normalize_face(frame)
				
				# show landmarks, after normalization
				#show_landmarks(inputdir,first_frame,current_frame)
					
				feature_vector = []
				for flm,clm in itertools.izip(first_frame,current_frame):
					feature_vector.append(clm[0]-flm[0])
					feature_vector.append(clm[1]-flm[1])
				feature_vector.extend(create_appearance_features(img,frame,n_points,radius))
				feature_vectors.append(feature_vector)
				
			expression_data.append(feature_vectors)
			expression_data.append(images_data)

			test_data.append(expression_data)

	elif test_method=='videos':
		# test directory
		test_data = []
		testdir = inputdir + '/videos'
		tests = [name for name in os.listdir(testdir) if os.path.isfile(os.path.join(testdir, name))]
		temp_tests = []
		for test in tests:
			if test.find('.avi')>0 or test.find('.mp4')>0 or test.find('.mpeg')>0:
				temp_tests.append(test)
		tests = temp_tests
		
		for test in tests:
		
			# load images and detect kandmarks
			expression_data = []
			landmarks_data = []		
			images_data = []
			expressiondir = testdir + '/' + test

			# read test video
			input_video = cv2.VideoCapture(expressiondir)
			
			# read test video frames
			frame_counter = 0
			while(input_video.isOpened()):
			
				# continue, while end of video has not reached
				ret, img = input_video.read()
				if not ret:
					break
				
				if frame_counter%5 == 0:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					images_data.append(img)
					dets = face_detector(img, 1)
					if len(dets)>0:
						d = dets[0]
						parts = landmark_detector(img, d).parts()
						lms_data = []
						landmarks = np.matrix([[p.x, p.y] for p in parts])
						for lm in landmarks:
							lm_data = []
							lm_data.append(int(lm.tolist()[0][0]))
							lm_data.append(int(lm.tolist()[0][1]))
							lms_data.append(lm_data)
						landmarks_data.append(lms_data)
					else:
						print 'failed to detect landmarks!'
				frame_counter = frame_counter + 1
					
			# first frame of video
			first_frame = landmarks_data[0]
			
			# normalize first frame
			first_frame = normalize_face(first_frame)
				
			# next frames of video
			feature_vectors = []
			for img, frame in itertools.izip(images_data, landmarks_data):
			
				# normalize current frame
				current_frame = normalize_face(frame)
				
				# show landmarks, after normalization
				#show_landmarks(inputdir,first_frame,current_frame)
					
				feature_vector = []
				for flm,clm in itertools.izip(first_frame,current_frame):
					feature_vector.append(clm[0]-flm[0])
					feature_vector.append(clm[1]-flm[1])
				feature_vector.extend(create_appearance_features(img,frame,n_points,radius))
				feature_vectors.append(feature_vector)
				
			expression_data.append(feature_vectors)
			expression_data.append(images_data)

			test_data.append(expression_data)
	
	return test_data

# create test features
def create_test_feature_vectors(data):
	
	temp_features = []
	for test_data in data:
		temp_features.append(test_data[0])
	features = temp_features
	
	return features
	
# perform a voting from neighbour frames for current frame label
def refinment(labels):

	temp_labels = []
	for test_label in labels:
		temp_test_label = []
		for k,lb in enumerate(test_label):
			if k>1 and k<len(test_label)-2:
				list = [test_label[k-1], test_label[k], test_label[k+1]]
				lb = Counter(list).most_common(1)[0][0]
			temp_test_label.append(lb)
		temp_labels.append(temp_test_label)
	labels = temp_labels
	
	return labels
	
# save classification result
def save_result(data, labels, outputdir, fourcc, frame_rate, resolution, test_method):

	print 'Saving Result...'
	
	i = 1
	if test_method=='images':
		videosdir = outputdir + '/images'
	elif test_method=='videos':
		videosdir = outputdir + '/videos'
	else:
		print 'Wrong Method!'
		
	for test_data, test_label in itertools.izip(data, labels):
	
		imagedir = os.path.join(videosdir,str(i))
		if not os.path.exists(imagedir):
			os.mkdir(imagedir)
		
		# write output
		test_images = test_data[1]
		j = 1
		for img,lb in itertools.izip(test_images, test_label):
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
			
			if lb==0:
				label = 'Neutral'
			elif lb==1:
				label = 'Angry'
			elif lb==2:
				label = 'Contempt'
			elif lb==3:
				label = 'Disgust'
			elif lb==4:
				label = 'Fear'
			elif lb==5:
				label = 'Happy'
			elif lb==6:
				label = 'Sadness'
			elif lb==7:
				label = 'Surprise'
				
			# add emoji to the image
			emojidir = inputdir+'/emoji'
			emoji = cv2.imread(os.path.join(emojidir, label+'.png'))
			emoji = cv2.resize(emoji, None, fx=0.15, fy=0.15)
			offset_x = 25
			offset_y = 50
			rows,cols,channels = emoji.shape
			roi = img[offset_y:offset_y+rows, offset_x:offset_x+cols ]
			emojigray = cv2.cvtColor(emoji,cv2.COLOR_BGR2GRAY)
			ret, mask = cv2.threshold(emojigray, 250, 255, cv2.THRESH_BINARY)
			mask = cv2.bitwise_not(mask)
			mask_inv = cv2.bitwise_not(mask)
			img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
			emoji_fg = cv2.bitwise_and(emoji,emoji,mask = mask)
			dst = cv2.add(img_bg,emoji_fg)
			img[offset_y:offset_y+rows, offset_x:offset_x+cols ] = dst

			# write expression on the image	
			cv2.putText(img, label, (25,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0,255,0))
		
			# save output image file
			cv2.imwrite(os.path.join(imagedir, 'image'+str(j)+'.jpg'), img)
			#cv2.imshow('img',img)
			#cv2.waitKey(400)
			
			j = j + 1
			
		i = i + 1
		
	return
	
# perform test on images data format
def test(training_data, outputdir, fourcc, frame_rate, resolution, test_method):

	test_data = load_test_data(inputdir, face_detector, landmark_detector, test_method)
	
	training_features, training_labels = create_feature_vectors(training_data,'train')
		
	test_features = create_test_feature_vectors(test_data)
	
	labels = []
	for features in test_features:
		predicted_labels = classification(classifier, training_features, training_labels, features)
		labels.append(predicted_labels)
		
	labels = refinment(labels)
	
	save_result(test_data, labels, outputdir, fourcc, frame_rate, resolution, test_method)
	
	return
	
# program main
if __name__ == '__main__':
	
	inputdir, outputdir, datasetdir, reload_dataset, face_detector, landmark_detector, region_width, n_points, radius, cross_validation_method, classifier, fourcc, frame_rate, resolution, test_method = init()
	
	dataset_data = load_dataset_features(inputdir, datasetdir, reload_dataset, face_detector, landmark_detector, region_width, n_points, radius)
	
	if test_method=='cross':
		cross_validation(dataset_data, cross_validation_method, classifier)
		
	elif test_method=='images' or test_method=='videos':
		test(dataset_data, outputdir, fourcc, frame_rate, resolution, test_method)
	
	cv2.destroyAllWindows()
	