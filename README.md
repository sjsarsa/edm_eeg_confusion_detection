# Educational Data Mining EEG Confusion Detection
This is a project for the Helsinki University course on Educational Data Mining. 

## Our goal
We follow the example of the paper [Confused or not confused?](https://dl.acm.org/citation.cfm?id=3107513) by Zhaoheng Ni et al and try to create a recurrent neural network (RNN) model capable of inferring students' confusion i.e. brain fog during watching online lecture videos. 

The authors show in the paper that data obtained from a single-channel EEG headset is enough to classify binary confusion with reasonable accuracy. Our first aim is to replicate the work done in the paper, since the [data](#data) is openly available.

Furthermore, we test the effect of adding additional information of the videos themselves. We extract information of the video by adding raw image data and sentence vectors produced from subtitle captions.

## Data

The data we use is hosted on the website [Kaggle](https://www.kaggle.com/wanghaohan/confused-eeg/home) and it is provided by the authors of [Using EEG to Improve Massive Open Online Courses Feedback Interaction](http://www.cs.cmu.edu/~kkchang/paper/WangEtAl.2013.AIED.EEG-MOOC.pdf).

## Classifying confusion
We run multiple models readily implemented for use in Python's Scikit-Learn library in order to have a good baseline for our
LSTM model. We compute accuracy as our primary metric, since it is the only one used in the papers we base our study on.
Additionally we provide [F1-score](https://en.wikipedia.org/wiki/F1_score) and [ROC-AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) to gain more insight on the performance of the models.

Models used are as follows:
  * (Naive) Predicts all labels as zero regardless of input
  * (Logreg) Logistic Regression 
  * (Ptron) Perceptron
  * (KNN) K-nearest neighbours
  * (GNB) Naive Bayes with Gaussian priors
  * (BNB) Naive Bayes with Bernoulli priors
  * (RF) Random Forest
  * (GBT) Gradient Boosted Decision Trees
  * (MLP2) Multilayer Perceptron with 2 hidden layers
  * (MLP3) Multilayer Perceptron with 3 hidden layers
  * (SVC Linear) Support Vector Machine with linear kernel
  
The models are all tested by 5-fold cross-validation and the results can be seen in the image below.
![Whoops!](https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_original_data.png)

From the plot we can see that the best models here (GBT, SVC) achieve near 70% accuracy.
When compared to Zhuoheng et al's reported results (picture below), the SVC performs similarly but interestingly our KNN performs clearly better than the one in the paper. The paper doesn't report KNN configuration, but their [code](https://github.com/nateanl/EEG_Classification/blob/master/KNN.py) uses sklearn default methods which might explain the differences (TODO: find out).

### Trying to reproduce Confused or not confused? -paper's results 

## Leveraging subtitle information
### Generating subtitle imbeddings
### Applying subtitle vectors to models

## Leveraging image data

## Classifying predefined difficulty

## Conclusion
