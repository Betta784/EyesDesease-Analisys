# EyesDesease-Analisys
This code is used for images deseases classification 

It is composed of these main parts:

Importing data
Splitting data (0.8)
Resampling training set 
Creating hog features
PCA application (on train set)
Applying KMeans clustering, VISUALIZATION
Applying KNN, VISUALIZATION, accurancy calculating
Applying RandomForestClassifier, VISUALIZATION, accurancy calculating
Applying SVN, VISUALIZATION, accurancy calculating, SVN
Applying DecisionTreeClassifier, VISUALIZATION, accurancy calculating
ROC curve visualization for KNN, RandomForestClassifier, DecisionTreeClassifier
For each process: COMPUTATIONAL TIME CALCULATIING

You will need to have matplotlib, scikit-learn, pandas, cv2, glob, skimage, sklearn, mpl_toolkits, seaborn, imbalanced-learn  and numpy installed.

The code was tested on Python 3.13.3 on Windows.

Usage
Run ldlPark.py. In the same folder you need the dataset folder named 'dataset'.
See here to download the dataset:
https://www.kaggle.com/code/mahnazarjmand/eye-diseases-classification
