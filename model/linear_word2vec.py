#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Activation, CuDNNLSTM, Bidirectional, Dropout
from keras.optimizers import RMSprop
import numpy as np
import os
import pandas as pd
import MeCab
import re
from sklearn.utils import  shuffle
from make_numpy import make_numpy
from copy import deepcopy
import json
import csv
import MeCab

class Tags_word2vec_module:
    # predictionとtraining共有の変数を設定
    batch_size = 512  # Batch size for training.
    epochs = 1000  # Number of epochs to train for.
    latent_dim = 200  # Latent dimensionality of the encoding space.
    latent_dim2 = 200
    num_samples = 120000  # Number of samples to train on.
    max_content = 300
    max_sentence_length = 300

    # リストを作成
    input_texts = list() # parsed sentences
    target_texts = list() # parsed tags
    input_words = set() # each character of parsed sentences
    target_chars = set() # each character of parsed tags
    parsed_sentence_length = list()
    parsed_tag_length = list()
    max_tag_length = 7
    max_char_length = 20000
    max_texts = -1

    def prepare(self):
        print("prepare is called")

        mecab_arg = '-F\s%f[6] -U\s%m -E\\n'
        sub_format = u"(https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+)|[!-~]|[︰-＠]|[)(＜＞＆。＿％『→；〜．「」、↓♪■］［∴・…]"
        tagger = MeCab.Tagger(str(mecab_arg))
        sentences, tag_names, self.target_texts = make_numpy()
        for sentence in sentences:
            if isinstance(sentence, str):
                # 文章の形態素解析
                sentence_formatted = re.sub(sub_format, "", sentence) # delete the redundant strings from the sentence
                sentence_parsed = tagger.parse(sentence_formatted).split() # split the sentence by comma which were separated
                # print(tagger.parse(sentence))
                self.parsed_sentence_length.append(len(sentence_parsed)) # add the length of words to the list
                input_text = sentence_parsed[:self.max_sentence_length] # 長さ調整
                self.input_texts.append(input_text) #listのlist
                for char in sentence_parsed:
                    self.input_words.add(char)

        #shuffle/sort the list_sentences_1
        self.input_texts, self.target_texts = shuffle(self.input_texts, self.target_texts)
        self.input_words = sorted(list(self.input_words))
        self.input_words.append('UKW')

        self.num_encoder_tokens = len(self.input_words) + 1
        #self.num_decoder_tokens = len(set(self.target_texts)) + 1
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts]) #TODO 101でOK?
        self.input_words_indexed = dict([(char, i) for i, char in enumerate(self.input_words, start=1)])
        #self.target_token_indexed = dict([(text, i) for i, text in enumerate(set(self.target_texts), start=1)])
        self.input_words_indexed['PAD'] = 0
        #self.target_token_indexed['PAD'] = 0

        self.num_train = int(len(self.input_texts)*0.9)
        self.num_valid = len(self.input_texts) - self.num_train

    def generater(self, isValid):
        print("generater is called")

        # 学習に使用するencoderとdecoderのデータを作成する。
        self.input_sequence = np.zeros((self.batch_size, self.max_encoder_seq_length), dtype='int32')
        self.target_sequence = np.zeros((self.batch_size, 100))
        ukw_input_index = self.input_words_indexed['UKW']

        # make the numbers of validation and training sets
        num = int(len(self.input_texts)*0.9)
        if isValid:
            x, y = self.input_texts[num:], self.target_texts[num:]
        else:
            x, y = self.input_texts[0 : num], self.target_texts[0 : num]

        # Manufacture the generators
        while (True):
            for i, input_text in enumerate(x):
                count = i % self.batch_size
                for j, word in enumerate(input_text):
                    if self.input_words_indexed.get(word, 0) > 0:
                        self.input_sequence[count, j] = self.input_words_indexed[word]
                    else:
                        self.input_sequence[count, j] = unk_input_index # change: int()のキャストを削除

                if count == self.batch_size - 1:
                    turn = int(i/self.batch_size) # change: times of batch loop
                    self.target_sequence = y[turn*self.batch_size : (turn + 1)*self.batch_size]
                    X = deepcopy(self.input_sequence)
                    Y = deepcopy(self.target_sequence)
                    self.input_sequence = np.zeros((self.batch_size, self.max_encoder_seq_length), dtype='int32')
                    self.target_sequence = np.zeros((self.batch_size, 100), dtype='float64')
                    yield  X, Y

    def rnn_model(self):

        print("model is called")

        '''make RNN model'''
        input_tensor = Input(shape=(self.max_encoder_seq_length,))
        inputs2 = Embedding(self.num_encoder_tokens, 50)(input_tensor)
        outputs1 = Bidirectional(LSTM(self.latent_dim, return_sequences=True))(inputs2)
        outputs2 = Bidirectional(LSTM(self.latent_dim2, return_sequences=False, return_state=False))(outputs1)
        dense1 = Dense(units=500, activation="relu")(outputs2)
        dense1 = Dropout(0.5)(dense1)
        dense2 = Dense(units=1000, activation="relu")(dense1)
        dense2 = Dropout(0.5)(dense2)
        dense3 = Dense(units=100)(dense2)

        self.model = Model(input_tensor, dense3)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.summary()

    def train_model(self):

        print("train is called")

        # コールバック関数を指定して学習させる。
        checkpoint = keras.callbacks.ModelCheckpoint('checkpoint_att_data_kirei_all.hdf5')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=20, verbose=0, mode='auto')
        reduce_lro = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        callbacks = [checkpoint, early_stop, reduce_lro]
        gen_train = self.generater(False)
        gen_valid = self.generater(True)
        self.model.fit_generator(gen_train,
                                 steps_per_epoch= self.num_train//self.batch_size,
                                 epochs=self.epochs,
                                 callbacks=callbacks,
                                 validation_data=gen_valid,
                                 validation_steps=self.num_valid//self.batch_size)
        '''
        self.model.save('checkpoint_att_data_kirei_nogpu_all.hdf5')
        with file_io.FileIO('checkpoint_att_data_kirei_nogpu_all.hdf5', 'r') as reader:
            with file_io.FileIO(os.path.join(self.model_path, 'checkpoint_att_data_kirei_nogpu_all.hdf5'), 'w+') as writer:
                writer.write(reader.read())
        '''
