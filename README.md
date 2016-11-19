# Face-Detection
The sliding window model is conceptually simple: independently classify all image patches as being object or non-object. Sliding window classification is the dominant paradigm in object detection and for one object category in particular -- faces -- it is one of the most noticeable successes of computer vision. For example, modern cameras and photo organization tools have prominent face detection capabilities. 

In this project, we perform the task of Face Detection. Face detection is used in various imagine tasks, and even in common objects like camera as shown in the image. The main method used is generation of Histogram Oriented Gradient features using Sliding window, as described in Dalal-Triggs paper. For classifying the sliding windows, we train a linear SVM.

In addition to the base implementation, we also implement various add-on techniques to observe and contrast the performance. The performance is compared based on Average Precision, ROC curve and values of confusion matrix (True Positives, False Negatives etc.). We implement the following extra techniques:

Implementation of HOG feature descriptor from scratch
 - Negative Hard Mining
 - Additional Dataset of positive training images
 - Implementation of Dense Neural Network in Python (using Keras) for classification.

The base implementation is quite fast. One complete run can take anywhere from 25s to 2min depending on parameters like HOG cell size, threshold for SVM classification, etc. The training images is the Caltech face dataset, cropped to 36 x 36 sized faces. The non-face training images are random (36 x 36) sized crops of non-face scene images.

