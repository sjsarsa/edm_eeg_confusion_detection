# Educational Data Mining EEG Confusion Detection
This is a project for the Helsinki University course on Educational Data Mining. 

## Our goal
We follow the example of the paper [Confused or not confused?](https://dl.acm.org/citation.cfm?id=3107513) and try to create a recurrent neural network (RNN) model capable of inferring students' confusion i.e. brain fog during watching online lecture videos. 

The authors in the aforementioned paper show that data obtained from a single-channel EEG headset is enough to classify binary confusion with reasonable accuracy. Our first aim is to replicate the work done in the paper, since the [data](#data) is openly available.

Furthermore, we test the effect of adding additional information of the videos themselves. We extract information of the video by adding raw image data and sentence vectors produced from subtitle captions.

## Data

The data we use is hosted on the website [Kaggle](https://www.kaggle.com/wanghaohan/confused-eeg/home) and it is provided by the authors of [Using EEG to Improve Massive Open Online Courses Feedback Interaction](http://www.cs.cmu.edu/~kkchang/paper/WangEtAl.2013.AIED.EEG-MOOC.pdf).

## Classifying confusion

## Leveraging subtitle information
### Generating subtitle imbeddings
### Applying subtitle vectors to models

## Leveraging image data

## Classifying predefined difficulty

## Conclusion
