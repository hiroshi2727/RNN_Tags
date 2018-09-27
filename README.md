# RNN_tags
## Content of this reposiotory
My internship project of natural language processing using deep learning.
## Flow of software function
### Learning
1. Import csv file of Japanese sentences with several tags as input.
2. Parse the sentences to words by space so that it can be treated as the input of RNN model (also called "Wakati").
3. Make a word2vec model based on sentences, which convert word to 100-200 dimention vectors.
4. Make a RNN model composed of LSTM, Dense, and Dropout layers for learning.
5. Flow the input sentences and the label of vectorized tag words into RNN model using generator for learning.
### Prediction
1. Import csv file of Japanese sentences as input.
2. Make RNN model for prediction importing the learned weights into model.
3. Predict the tags for each sentence by taking the test sentences as input.
## Folders and files
### Folders
1. clustering: vector clustering for linear prediction
2. preprocessing: preprocess the files before learning
3. model: actural learning and prediction files with various models
### Files in model folder
1. categorical_onehot.py: Use one-hot vectors as inputs and categorical model for predicting.
2. categorical_multihot.py: Use multi-hot vectors as inputs and categorical model for predicting.
3. categorical_seq2seq.py: Use one-hot vectors as inputs and sequence to sequence model for predicting.
4. linear_word2vec.py: Use vectorized word by word2vec as inputs and linear regression model for predicting.
