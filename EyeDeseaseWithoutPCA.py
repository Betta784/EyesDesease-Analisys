import glob
import numpy as np # linear algebra
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.utils import resample
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay,accuracy_score
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.mplot3d import Axes3D # If you have matplotlib installed, you should be able to import mpl_toolkits
from sklearn.cluster import KMeans
import seaborn as sns

#Servono:
#python -m pip install -U pip
#python -m pip install -U matplotlib
#python -m pip install -U numpy
#python -m pip install -U opencv-contrib-python
#python -m pip install -U pandas
#python -m pip install -U scikit-image
#python -m pip install -U imbalanced-learn 
#python -m pip install -U seaborn


#DATASET import

const_img_w=128;
const_img_h=128;
const_figure_W=1200;
const_figure_H=720;

finalClasses=[0,1,2,3];
finalLabels=["normal","glaucoma","diabetic_retinopathy","cataract"];
constDirPath="./dataset/";
txtfiles = [];

pxInInches = 1/plt.rcParams['figure.dpi'];  # pixel in inches
		
startTimeInSeconds = time.time();
figHOG, axsHOG = plt.subplots(2, 2,figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot HOG
figCM, axsCM = plt.subplots(2, 2,figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot dei ConfusionMatrix
figROC, axsROC = plt.subplots(figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot ROC

images=[];
def get_imgs(labels,classes,imagesToBeFilled,resW,resH):
	data = pd.DataFrame({"labelName": [],"labelValue": [],"image":[]});
	for i in range(len(labels)):
		strFilePath=constDirPath+labels[i]+"/*.*";
		for file in glob.glob(strFilePath):
			imageOrig = cv2.imread(file); #use the second argument or (flag value) zero that specifies the image is to be read in grayscale mode
			h, w,c = imageOrig.shape;
			if h==w:
				down_points = (resW, resH)
				image = cv2.resize(imageOrig, down_points, interpolation= cv2.INTER_LINEAR)
				new_row = {"labelName": str(labels[i]), "labelValue": str(classes[i]), "image": image};
				data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True);
			else:
				print(f"Not squared image Dimension {file} ");
	return data;

data=get_imgs(finalLabels,finalClasses,images,const_img_w,const_img_h);
print(f"OriginalData {data['labelName'].value_counts()}");
print(f"data {data.shape} ");



#SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(data, data["labelValue"], random_state=0, train_size = 0.8)
print(f"X_train {X_train.shape} y_train {y_train.shape} X_test {X_test.shape} y_test {y_test.shape} ");
print(f"Afer split Training:{X_train['labelName'].value_counts()} Test:{X_test['labelName'].value_counts()}"); 


#RESAMPLING TRAINING SET
#oversampled trainingset,Random Oversampling (Random oversampling involves duplicating random instances from the minority class until it is balanced with the majority class.). 
#Non sono riuscita con SMote. SMOTE generates synthetic instances of the minority class based on the existing data, reducing the risk of overfitting
minority_class1 = X_train[X_train['labelName'] == 'normal'];
minority_class2 = X_train[X_train['labelName'] == 'glaucoma'];
minority_class3 = X_train[X_train['labelName'] == 'cataract'];
majority_class = X_train[X_train['labelName'] == 'diabetic_retinopathy']
# Upsample the minority class
minority_upsampled1 = resample(minority_class1, replace=True, n_samples=len(majority_class), random_state=42);
minority_upsampled2 = resample(minority_class2, replace=True, n_samples=len(majority_class), random_state=42);
minority_upsampled3 = resample(minority_class3, replace=True, n_samples=len(majority_class), random_state=42);
# Combine the upsampled minority class with the majority class
balanced_data = pd.concat([majority_class, minority_upsampled1,minority_upsampled2,minority_upsampled3]);
print(f"Afer Training Oversampled {balanced_data['labelName'].value_counts()}!");   
y_train_balanced=balanced_data['labelValue']; 

print(f"balanced_data {balanced_data.shape} y_train_balanced {y_train_balanced.shape} X_test {X_test.shape} y_test {y_test.shape} ");
 

#HOG IMPLEMENTATION
def myOrig_hog(data_in):
	#creating hog features
	features = [];
	for i in range(len(data_in)):
		img=data_in.iloc[i]['image'];
		
		hogDesc, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True,feature_vector=True,channel_axis=2  ); #channel_axis=2 è nel caso di multichannel image (penso a colori),  If None, the image is assumed to be a grayscale (single channel) image.Otherwise, this parameter indicates which axis of the array corresponds to channels.
		#orientations: Number of orientation bins.

		hogDescriptorRounded=np.round(hogDesc, 2);#2 decimali solo		
		features.append(hogDesc);

		if i==2000: # stampo per esempio histogramma di uno solo
			#print ('HOG Descriptor has shape:', hog_descriptor_withOpenCV.shape)
			#print(hogDesc.shape);#viene 15, 15, 2, 2, 9 cioè un primo vettore di 15 (gruppi h),vettore di 15(gruppiv), 2 (cell H), 2 (cell V), 9 (che sono i bin)
			unique, counts = np.unique(hogDescriptorRounded, return_counts=True);
			dictCount=dict(zip(unique, counts))
			print(dictCount);
			axsHOG[0,0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY));
			axsHOG[0,0].set_title('OriginalImage:'+data_in.iloc[i]['labelName']);
			axsHOG[0,1].imshow(hog_image, cmap="gray");
			axsHOG[0,1].set_title('HOG');
			axsHOG[1,1].set_title('HOG features');
			axsHOG[1,1].bar([ str(i) for i in dictCount.keys()], dictCount.values(), color='b');
			axsHOG[1,1].set_xticks(axsHOG[1,1].get_xticks()[::10]);#metto una label sull'asse x ogni 10	
	return (np.asarray(features));
	

train_hog_fts=myOrig_hog(balanced_data);
test_hog_fts=myOrig_hog(X_test);
print ('train_hog_fts shape:', train_hog_fts.shape);
print ('test_hog_fts shape:', test_hog_fts.shape);
print ('y_train_balanced shape:', y_train_balanced.shape);
print ('y_test shape:', y_test.shape);



# Scale data before applying PCA
std_scaler = StandardScaler();

#questo trasforma per media 0 e varianza 1 (ma penso che non devo farlo, perchè HOG già lo fa)
std_scaler.fit(train_hog_fts);#fit on training only (https://builtin.com/machine-learning/pca-in-python)
train_hog_fts=std_scaler.transform(train_hog_fts);
test_hog_fts=std_scaler.transform(test_hog_fts);



#https://medium.com/@kumudtraveldiaries/metrics-in-random-forest-daf32de3ed6b
#Predict probabilities for ROC AUC  using predict_proba()
#For accuracy_score using predict()
#for multiclass ROC oneVsRest: https://scikit-learn.ru/stable/auto_examples/model_selection/plot_roc.html

#KNN 
knn = KNeighborsClassifier(metric='cosine', n_neighbors=18, weights='distance');
knn.fit(train_hog_fts, y_train_balanced);
predictionsKNN = knn.predict(test_hog_fts);#ndarray 

y_score = knn.predict_proba(test_hog_fts)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
fpr_KNN, tpr_KNN, _ = metrics.roc_curve(y_onehot_test[:, class_id], y_score[:, class_id]);
auc_KNN = round(metrics.roc_auc_score(y_onehot_test[:, class_id], y_score[:, class_id]), 4);

cmKNN = confusion_matrix(y_test, predictionsKNN);
dispKNN = ConfusionMatrixDisplay(confusion_matrix=cmKNN, display_labels=finalLabels);
dispKNN.plot(ax=axsCM[0,0],cmap=plt.cm.Blues,xticks_rotation=75);
print("KNN Confusion Matrix: \n", cmKNN);
print("\n");
print(classification_report(y_test, predictionsKNN));#The support is the number of occurrences of each class in y_true.
knn_model_acc = accuracy_score(y_test, predictionsKNN);
print("Accuracy of K Neighbors Classifier is: ", knn_model_acc);

#RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300);
rfc.fit(train_hog_fts, y_train_balanced);
predictionsRFC = rfc.predict(test_hog_fts);

y_score = rfc.predict_proba(test_hog_fts)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
fpr_RFC, tpr_RFC, _ = metrics.roc_curve(y_onehot_test[:, class_id],	y_score[:, class_id]);
auc_RFC = round(metrics.roc_auc_score(y_onehot_test[:, class_id],	y_score[:, class_id]), 4);

cmRFC = confusion_matrix(y_test, predictionsRFC);
dispRFC = ConfusionMatrixDisplay(confusion_matrix=cmRFC, display_labels=finalLabels);
dispRFC.plot(ax=axsCM[0,1],cmap=plt.cm.Blues,xticks_rotation=75);
print("RFC Confusion Matrix: \n", cmRFC);
print("\n");
print(classification_report(y_test, predictionsRFC));
rfc_acc = accuracy_score(y_test, predictionsRFC);
print("Accuracy of Random Forests Classifier is: ", rfc_acc);


#SVC, a specific implementation of SVM in Scikit-learn, is widely used for binary and multi-class classification tasks. While SVMs are inherently non-probabilistic, Scikit-learn provides a mechanism to extract probability estimates through the predict_proba() function.
#The predict_proba() function is designed to give the probability estimates for each class label in a classification task. This is particularly useful in applications where understanding the confidence of a prediction is as important as the prediction itself.
#https://www.geeksforgeeks.org/machine-learning/understanding-the-predictproba-function-in-scikit-learns-svc/
svc_classifier = SVC(kernel="rbf",probability=True); #possibility: kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’ ( radial basis function kernel)
svc_classifier.fit(train_hog_fts, y_train_balanced); # take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, and an array y of class labels (strings or integers), of shape (n_samples):
predictionsSVM = svc_classifier.predict(test_hog_fts);

y_score = svc_classifier.predict_proba(test_hog_fts)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
fpr_SVM, tpr_SVM, _ = metrics.roc_curve(y_onehot_test[:, class_id],	y_score[:, class_id]);
auc_SVM = round(metrics.roc_auc_score(y_onehot_test[:, class_id],	y_score[:, class_id]), 4);


cmSVM = confusion_matrix(y_test, predictionsSVM);
print("SVM Confusion Matrix: \n", cmSVM);
print("\n");
print(classification_report(y_test, predictionsSVM));
dispSVM = ConfusionMatrixDisplay(confusion_matrix=cmSVM, display_labels=finalLabels);
dispSVM.plot(ax=axsCM[1,0],cmap=plt.cm.Blues,xticks_rotation=75);
svc_acc = accuracy_score(y_test, predictionsSVM);
print("Accuracy of SVC Classifier is: ", svc_acc);

#DecisionTreeClassifier
dct_classifier = DecisionTreeClassifier();
dct_classifier.fit(train_hog_fts, y_train_balanced);
predictionsDCT = dct_classifier.predict(test_hog_fts);

y_score = dct_classifier.predict_proba(test_hog_fts)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
fpr_DCT, tpr_DCT, _ = metrics.roc_curve(y_onehot_test[:, class_id],	y_score[:, class_id]);
auc_DCT = round(metrics.roc_auc_score(y_onehot_test[:, class_id],	y_score[:, class_id]), 4);

cmDCT = confusion_matrix(y_test, predictionsDCT);
dispDCT = ConfusionMatrixDisplay(confusion_matrix=cmDCT, display_labels=finalLabels);
dispDCT.plot(ax=axsCM[1,1],cmap=plt.cm.Blues,xticks_rotation=75);
print("DecisionTree Confusion Matrix: \n", cmDCT);
print("\n");
print(classification_report(y_test, predictionsDCT));
DCT_acc = accuracy_score(y_test, predictionsDCT);
print("Accuracy of Decision Tree Classifier is: ", DCT_acc);


#PLOT ROC CURVE
axsROC.plot(fpr_KNN,tpr_KNN,label="KNN, AUC="+str(auc_KNN));
axsROC.plot(fpr_RFC,tpr_RFC,label="RFC, AUC="+str(auc_RFC));
axsROC.plot(fpr_SVM,tpr_SVM,label="SVM, AUC="+str(auc_SVM));
axsROC.plot(fpr_DCT,tpr_DCT,label="DCT, AUC="+str(auc_DCT));
axsROC.set_title('ROC');
axsROC.legend()

#COMPUTATIONAL TIME CALCULATION
elapsedTimeInSeconds = time.time()-startTimeInSeconds;
print("elapsedTimeInSeconds is: ", elapsedTimeInSeconds);

axsCM[0,0].set_title('KNN Confusion Matrix');
axsCM[0,1].set_title('Random Forests Confusion Matrix');
axsCM[1,0].set_title('SVM Confusion Matrix');
axsCM[1,1].set_title('Decision Tree Confusion Matrix');
figHOG.tight_layout();
figCM.tight_layout();
figROC.tight_layout();
plt.tight_layout();
plt.show();


