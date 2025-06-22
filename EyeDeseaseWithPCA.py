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
figOptimalFeatures, axsOptimalFeatures = plt.subplots(figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches) );#per il PCAfull per vedere le optimal feaures
figHOG, axsHOG = plt.subplots(2, 2,figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot HOG
figCM, axsCM = plt.subplots(2, 2,figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot dei ConfusionMatrix
figPCA, axsPCA = plt.subplots(2, 2,figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot PCA
#figPCALoading, axsPCALoading = plt.subplots(figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot PCA loading
figROC, axsROC = plt.subplots(figsize=(const_figure_W*pxInInches, const_figure_H*pxInInches));#per il plot ROC
# remove the subplots to be set as 3d projections
axsPCA[0,0].remove();
axsPCA[1,0].remove();
# add the subplots back as 3d projections; rows, cols and index are relative to width_ratios
axsPCA[0,0] = figPCA.add_subplot(2, 2, 1, projection='3d');#2 rows, 2 columns, position 1  (index is 1-based, ad esempio se  3 rows, 4 columns, index=9 è il primo della terza riga)
axsPCA[1,0] = figPCA.add_subplot(2, 2, 3, projection='3d');


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
		#questo con opencv 
		img_gray_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
		h, w,c = img.shape;
		if i%200==0:
			print(f"img[{i}] w={w} h={h}"); 
		# Specify the parameters for our HOG descriptor
		win_size = img_gray_opencv.shape;
		cell_size = (8, 8);
		block_size = (16, 16);
		block_stride = (8, 8);
		num_bins = 9;
		# Set the parameters of the HOG descriptor using the variables defined above
		hog_withOpenCV = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins);
		#hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins);
		# Compute the HOG Descriptor for the gray scale image
		hog_descriptor_withOpenCV = hog_withOpenCV.compute(img_gray_opencv);
		#print ('HOG Descriptor has shape:', hog_descriptor_withOpenCV.shape);
		
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


#PCA 

####################################Determining the Optimal Number of PCA Components (si vede solo visivamente, dopo calcolo il numero di features)
# Fit PCA to the data without reducing dimensions and compute the explained variance ratio
pca_full = PCA().fit(train_hog_fts)
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Create a plot for cumulative explained variance

axsOptimalFeatures.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
axsOptimalFeatures.set_title('Cumulative Explained Variance')
axsOptimalFeatures.set_xlabel('Number of components')
axsOptimalFeatures.set_ylabel('Cumulative explained variance')
axsOptimalFeatures.axhline(y=0.85, color='r', linestyle='-')  # 85% variance line
axsOptimalFeatures.text(0.5, 0.8, '85% cut-off threshold', color = 'red', fontsize=16);
#############################################################









#vedi https://drlee.io/the-ultimate-step-by-step-guide-to-data-mining-with-pca-and-kmeans-83a2bcfdba7d
#scelgo n_components=3 (per ottenere i clusters in 3D)
pcaForPlot = PCA(n_components=3)
pcaForPlot_result = pcaForPlot.fit_transform(train_hog_fts)
axsPCA[0,0].scatter(pcaForPlot_result[:, 0], pcaForPlot_result[:, 1], pcaForPlot_result[:, 2])
axsPCA[0,0].set_xlabel('PC1')
axsPCA[0,0].set_ylabel('PC2')
axsPCA[0,0].set_zlabel('PC3')
explained_varianceForPlot = pcaForPlot.explained_variance_;
explained_variance_ratioForPlot = pcaForPlot.explained_variance_ratio_; 

grid_explained_varianceForPlot = np.arange(1, 3 + 1);
# Explained variance
axsPCA[0,1].bar(grid_explained_varianceForPlot, explained_variance_ratioForPlot);
axsPCA[0,1].set(  xlabel="Component", title="% Explained Variance "+str(3), ylim=(0.0, 1.0) );

#KMeans is a popular unsupervised learning technique used to identify clusters in data
#Clustering on PCA-transformed data can reveal patterns and groupings that might not be obvious in the high-dimensional original data
n_clusters = len(finalClasses);# appropriate number of clusters (in base alle 4 classi di immagini)

# Applying KMeans clustering
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_pca.fit(pcaForPlot_result);

# The cluster labels for each data point
cluster_labels = kmeans_pca.labels_
axsPCA[1,0].scatter(pcaForPlot_result[:, 0], pcaForPlot_result[:, 1], pcaForPlot_result[:, 2], c=cluster_labels)
axsPCA[1,0].set_xlabel('PC1');
axsPCA[1,0].set_ylabel('PC2');
axsPCA[1,0].set_zlabel('PC3');
axsPCA[1,0].set_title('3D KMeans Clustering on PCA Results');

#PCA loadings are coefficients that describe how each principal component is a combination of the original features. They help in understanding the nature of the principal components — whether they represent a particular group of features or a specific pattern in the data.

pca_components_forPlot = pcaForPlot.components_;# Get the PCA components (loadings)
print(pca_components_forPlot);
print(f"pca_components_forPlot: {pca_components_forPlot}!");
print(f"pcaForPlot.n_components_: {pcaForPlot.n_components_}!");

# Create a DataFrame for better visualization and analysis
pcaForPlot_loadings_df = pd.DataFrame(pca_components_forPlot, index=[f'PC{i+1}' for i in range(pcaForPlot.n_components_)])

# Heatmap of the loadings
#sns.heatmap(pcaForPlot_loadings_df, cmap="YlGnBu", annot=True,ax=axsPCALoading);
#axsPCALoading.set_title('PCA Loadings');





pca = PCA(0.85);#0.85 to achieve 85% variance. Torna n_components_ (numero di features per cui si raggiunge l'85% della varianza)
pca.fit(train_hog_fts);#fit on training only pca.fit(np.concatenate([train_hog_fts, test_hog_fts])); vedi https://builtin.com/machine-learning/pca-in-python

explained_variance = pca.explained_variance_;
explained_variance_ratio = pca.explained_variance_ratio_;
n_features_reduced=pca.n_components_;
print(f"n_features_reduced: {n_features_reduced}!");

grid_explained_variance = np.arange(1, n_features_reduced + 1);
# Explained variance
axsPCA[1,1].bar(grid_explained_variance, explained_variance_ratio);
axsPCA[1,1].set(  xlabel="Component", title="% Explained Variance "+str(n_features_reduced), ylim=(0.0, 0.20) );

#plt.plot(np.cumsum(explained_variance));
print(np.sum(explained_variance), np.sum(explained_variance_ratio));

train_hog_pca = pca.transform(train_hog_fts);
test_hog_pca  = pca.transform(test_hog_fts);



std_scaler.fit(train_hog_pca);# fit on training only std_scaler.fit(np.concatenate([train_hog_pca, test_hog_pca])); vedi https://builtin.com/machine-learning/pca-in-python
train_hog_pca = std_scaler.transform(train_hog_pca);
test_hog_pca = std_scaler.transform(test_hog_pca);
#queste le devo passare ai vari classificatori



print (y_test);
#print(type(y_test)) is a panda series
print ("aftyer y_test");
normalVsOther_y_test=y_test.replace(['0','1','2', '3'],[0,1,1,1]);
print (normalVsOther_y_test);
print ("aftyer normalVsOther_y_test");#ha n righe ed una colonna


#https://medium.com/@kumudtraveldiaries/metrics-in-random-forest-daf32de3ed6b
#Predict probabilities for ROC AUC  using predict_proba()
#For accuracy_score using predict()
#for multiclass ROC oneVsRest: https://scikit-learn.ru/stable/auto_examples/model_selection/plot_roc.html

#KNN 
knn = KNeighborsClassifier(metric='cosine', n_neighbors=18, weights='distance');
knn.fit(train_hog_pca, y_train_balanced);
predictionsKNN = knn.predict(test_hog_pca);#ndarray 

y_score = knn.predict_proba(test_hog_pca)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
print("class_id: \n", class_id);
display = RocCurveDisplay.from_predictions(
	y_onehot_test[:, class_id],
	y_score[:, class_id],
	name=f"Normal vs the rest",
	color="darkorange",
	plot_chance_level=True,
)
_ = display.ax_.set(
	xlabel="False Positive Rate",
	ylabel="True Positive Rate",
	title="KNN One-vs-Rest ROC curves:\nNormal vs Other",
)

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
rfc.fit(train_hog_pca, y_train_balanced);
predictionsRFC = rfc.predict(test_hog_pca);

y_score = rfc.predict_proba(test_hog_pca)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
print("class_id: \n", class_id);
display = RocCurveDisplay.from_predictions(
	y_onehot_test[:, class_id],
	y_score[:, class_id],
	name=f"Normal vs the rest",
	color="darkorange",
	plot_chance_level=True,
)
_ = display.ax_.set(
	xlabel="False Positive Rate",
	ylabel="True Positive Rate",
	title="RandomForest One-vs-Rest ROC curves:\nNormal vs Other",
)

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
svc_classifier.fit(train_hog_pca, y_train_balanced); # take as input two arrays: an array X of shape (n_samples, n_features) holding the training samples, and an array y of class labels (strings or integers), of shape (n_samples):
predictionsSVM = svc_classifier.predict(test_hog_pca);

y_score = svc_classifier.predict_proba(test_hog_pca)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
print("class_id: \n", class_id);
display = RocCurveDisplay.from_predictions(
	y_onehot_test[:, class_id],
	y_score[:, class_id],
	name=f"Normal vs the rest",
	color="darkorange",
	plot_chance_level=True,
)
_ = display.ax_.set(
	xlabel="False Positive Rate",
	ylabel="True Positive Rate",
	title="SVM One-vs-Rest ROC curves:\nNormal vs Other",
)

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
dct_classifier.fit(train_hog_pca, y_train_balanced);
predictionsDCT = dct_classifier.predict(test_hog_pca);

y_score = dct_classifier.predict_proba(test_hog_pca)
label_binarizer = LabelBinarizer().fit(y_train_balanced)
y_onehot_test = label_binarizer.transform(y_test)
class_of_interest = '0';
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
print("class_id: \n", class_id);
display = RocCurveDisplay.from_predictions(
	y_onehot_test[:, class_id],
	y_score[:, class_id],
	name=f"Normal vs the rest",
	color="darkorange",
	plot_chance_level=True,
)
_ = display.ax_.set(
	xlabel="False Positive Rate",
	ylabel="True Positive Rate",
	title="DecisionTree One-vs-Rest ROC curves:\nNormal vs Other",
)

cmDCT = confusion_matrix(y_test, predictionsDCT);
dispDCT = ConfusionMatrixDisplay(confusion_matrix=cmDCT, display_labels=finalLabels);
dispDCT.plot(ax=axsCM[1,1],cmap=plt.cm.Blues,xticks_rotation=75);
print("DecisionTree Confusion Matrix: \n", cmDCT);
print("\n");
print(classification_report(y_test, predictionsDCT));
DCT_acc = accuracy_score(y_test, predictionsDCT);
print("Accuracy of Decision Tree Classifier is: ", DCT_acc);



#COMPUTATIONAL TIME CALCULATION
elapsedTimeInSeconds = time.time()-startTimeInSeconds;
print("elapsedTimeInSeconds is: ", elapsedTimeInSeconds);

axsCM[0,0].set_title('KNN Confusion Matrix');
axsCM[0,1].set_title('Random Forests Confusion Matrix');
axsCM[1,0].set_title('SVM Confusion Matrix');
axsCM[1,1].set_title('Decision Tree Confusion Matrix');
figHOG.tight_layout();
figCM.tight_layout();
figPCA.tight_layout();
figOptimalFeatures.tight_layout();
#figPCALoading.tight_layout();
#figROC.tight_layout();
plt.tight_layout();
plt.show();


