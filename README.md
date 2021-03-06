# Facial Expression Classification

In this project, I've designed and implemented models for the classification of human facial expressions using the image and video data. The code has been written in Python based on the OpenCV library.

In phase I (i.e. classification using image data), a set of local descriptors have been detected based on the FAST, SURF, and MSER methods (the geometric distribution of landmarks has also been used for removing outlier points). Afterwards, the detected points have been represented by SIFT, BRIEF, BRISK, ORB, and FREAK descriptors. A Random Decision Forrest and a Multi-Layer Perceptron have then be used for classifying the expression types using these features. Two standard approaches, i.e. the DLIB package and the Active Appearance Model, have also been implemented for making the comparison with the state of the art models easier.

In phase II (i.e. classification using video data), a set of local descriptors have been detected in every frame of the video using the above-mentioned approach and normalized appropriately. Then two distinct types of features are extracted from these landmarks: 1- the relative changes in the geometric location of the landmarks, 2- local binary patterns around the landmarks. Feature selection has then been performed using an L-1 regularized Support Vector Machine and a Random Decision Forrest model is adopted for classification. An additional majority voting layer is also responsible for the refinement of the model's results by comparing the detected expressions in consecutive frames.

Please refer to the following paper for a recent review of the relevant literature:
Zhao, Xiaoming, and Shiqing Zhang. "A review on facial expression recognition: feature extraction and classification." IETE Technical Review 33.5 (2016): 505-517.
