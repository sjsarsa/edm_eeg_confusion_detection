# Educational Data Mining EEG Confusion Detection
This is a project for the Helsinki University course on Educational Data Mining. 

## Our goal
We follow the example of the paper [Confused or not confused?](https://dl.acm.org/citation.cfm?id=3107513) by Ni et al and try to create a recurrent neural network (RNN) model capable of inferring students' confusion i.e. brain fog during watching online lecture videos. 

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
  
The models are all tested by 5-fold cross-validation. We didn't however finetune parameters for the various models.

#### Results for cross-validating models:
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_original_data.png" height="250"/>

From the plot we can see that the best models here (GBT, SVC) achieve near 70% accuracy.
When compared to Zhuoheng et al's reported results (picture below), the SVC performs similarly but interestingly our KNN performs clearly better than the one in the paper. The paper doesn't report KNN configuration, but their [code](https://github.com/nateanl/EEG_Classification/blob/master/KNN.py) reveals that their input shape for KNN is (12811, 14) and output shape is (12811,). This means that the KNN is attempting to classify confusion for a whole video by just one interval. With proper input, i.e. 100 data points with 112 concatenated feature vectors resulting in shape (100, 112*12=1344), the KNN achieves accuracy of 70.9%. The small difference to our KNN result (68% accuracy) is likely due to differences in the implementation of cross-validation. We use our own cross-validation whereas the other code uses sklearn's ''[cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)'' 
.

#### Results from the paper "Confused or not confused?":
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_confused_paper_results.png" height="250"/>

### Attempting to reproduce "Confused or not confused?" -paper's results 
Our starting point was to create the model described in Ni et al's paper and try to improve on that. The model consists of  batch normalization layer, a 50 neuron Bidirectional Long-Short Term Memory (LSTM) Recurrent layer as described in the paper, and surprisingly a 112 neuron output layer. We first assumed that the model was supposed to predict 1 output for each subject-video pair that were our data points. However, we had difficulty producing stable results such as described in the paper. Our similar model's accuracy varied from 45% to 80% and even the cross-validation accuracy varied slightly while remaining around 60%. This indicates a high dependence for random initialization of the model and that a regular LSTM probably isn't the best choice for this task.

Luckily we have the [original LSTM model's code](https://github.com/nateanl/EEG_Classification/blob/master/EEG_LSTM.py) and we tested that also. First we simply ran it and got lower results than those in the paper. Then for proper validation, we removed the random seeds from the code and put the whole [cross-validation in a for-loop](https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/test_eeg_classification_lstm.ipynb). The accuracy was 61.3% , which is much closer to our attempts than what is in the paper. Thus, if there is no magic performed that has been omitted from the published code, it seems that the paper's results are not reproducible.
## Leveraging subtitle information
We decided to test the effect of adding subtitles to test if there was some semantic information in the videos that might help classifying confusion. For this we used YouTube's automatic captioning to get the subtitles as captions. We then mapped the captions to the videos' middle minute intervals so that it is in line with the kaggle EEG data. After that, we used pre-trained [ELMo word embedding model](https://github.com/HIT-SCIR/ELMoForManyLangs/) to convert the subtitles to word vectors and created subtitle vectors by averaging the genereted ELMo word vectors. Finally we added the subtitle vectors to the dataset and tested the models with the combined set.    

#### Results for adding subtitle vecs
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_subvecs.png" height="250"/>

It seems that the subtitle vectors didn't help at all in explaining the students' perceived confusion. For some reason percetron gets good results, but this might be due to chance as all other models don't seem affected. The reason for GBT's fall is simply reducing the number of trees drastically from 777 to 3. The GBT modification is done since training it with the vectors was extremely slow. 

The LSTM model is not shown on the graph, but we simply note that it did not seem to have any consistant effect on the model performance.

Additionally, we tried using PCA to reduce subtitle vector dimensions, since it [can be used to improve word vectors](https://ieeexplore.ieee.org/abstract/document/8500303).
#### Results for subtitle vecs' 12 principal components
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_pca_sub_vecs.png" height="250"/>

As can be seen, the subtitle vectors with reduced dimensionality don't seem to affect the models and now the GBT performs normally with its 777 estimator trees. And the LSTM yet again was unaffected. We also tried several dimension values for PCA, but the effect was minimal.

## Leveraging image data
Each video was sliced into frames for 0.5 second intervals (using the 1st and 15th frame of each second). These were then loaded into a numpy array as grayscale images.

## Classifying predefined difficulty
We stumbled across [Ali Mehmani's repository] where he had used a Neural Network that combined a 2-dimensional Convolution layer and two LSTM layers. Looking at his code we found that his model was predicting the pre-defined labels of the videos. Also, in comparison to the Ni et al Mehmani used pre-normalized data instead of batch normalization for the neural net.

We tested the model ourselves and truly, the model performed well on the pre-defined labels, however, not so much on the student-defined In addition, the model worked well only with zero padded data and not when the data was truncated to the minimum amount of intervals in the watched video data. In any case, the model's capability to infer predefined difficulty is rather interesting.

#### Mehmani model performance
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/mehmani_results.png" height="300"/> 
Mehamni used 10-fold cross-validation, but we got the similar results for our 5-fold cross-validation, accuracy: 0.780, F1: 0.828, and ROC-AUC: 0.809 

#### Baseline model performances for pre-defined difficulty truncated data
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_predefined_labels.png" height="250"/>

Results for Mehmani's model with truncated data for pre-defined labels: accuracy 0.550, F1: 0.536, and ROC-AUC: 0.558

#### Baseline model performances for pre-defined difficulty zero padded data
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_zero_pad_baseline_predefined.png" height="250"/>

The accuracy line here lies behind the ROC-AUC score and we can see that the Mehmani's model performance seems similar to decision tree based classifiers when predicting the pre-defined difficulty. Interestingly the way we even out the data has a huge efect on model performance as can be seen from comparing the plots with truncated and zero padded data. For predicting student-defined labels the effect was the opposite for all models.  

### Image Data
Using image data to classify predefined difficulty will overfit on the training set, unless significant pre-processing is done. This is because the videos defined as 'easy' are almost all videos from Khan Academy. The 'difficult' videos typically feature a physical lecturer in front of a blackboard. The initial intention was for the model to learn interesting features from the video data, however given the small size of the dataset and the clear visual distinctions between the two classes, it's likely to only learn that many grayscale values close to 0 (black) indicate 'easy' videos.

### Subtitle Vectors
Using the baseline models as a comparison, we found a couple of models worked well in utilizing the information encoded within the subtitle vectors. This indicates that the averaged ELMo embeddings carry enough semantic meaning to distuingish the difficult videos from non-difficult videos in this dataset.

#### Model performances for pre-defined difficulty with subtitle vectors
<img src="https://github.com/taikamurmeli/edm_eeg_confusion_detection/blob/master/plots_and_images/plot_subvecs_for_predefined_labels.png" height="250"/>
Also Mehmani's model achieved 1.0 for all scores regardless of how the intervals were evened out in the data.

## Conclusion
