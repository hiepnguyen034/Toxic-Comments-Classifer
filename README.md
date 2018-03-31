# Spam-Filter
Detect toxic/offensive messages using various classification techniques

The dataset used to train the models can be downloaded from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

This repository contains two approaches to deal with toxic comments.

The first approach is to use three classification models, including logistic regression, gardient boosting trees, and multilayer perceptron models to train each type of toxic comments (toxic, severe toxic,obscene, threat, insult, and identity hate) seperately. The resutls after testing the models are summarized in the Accuracy.csv file.

The second approach is to use LSTM and CNN models. Details can be found in the ipython notebook file.
