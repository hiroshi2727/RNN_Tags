#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Activation
import numpy as np
import os
import pandas as pd
import MeCab
import re
import json

class Tags_multihot:
    batch_size = 64  # Batch size for training.
    epochs = 10  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    latent_dim2 = 128
    max_encoder_seq_length = 300
    max_decoder_seq_length = 9 # アウトプット時、最大のタグの数
    # For prediction
    num_tags = 10
    num_words = 9
    # mecab
    mecab_arg = '-F\s%f[6] -U\s%m -E\\n'
    sub_format = u"(https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+)|[!-~]|[︰-＠]|[)(＜＞＆。＿％『→；〜．「」、↓♪■］［∴・…]"
    # リストを作成
    input_texts = list()
    target_texts = list()
    input_words = set()
    target_words = set()
    parsed_sentence_length = list()
    parsed_tag_length = list()

    def __init__(self, isTrain):
        self.isTrain = isTrain

    def mecab_for_train(self):
        print("macab is called")
        if not self.isTrain:
            print("error")
            return
        tagger = MeCab.Tagger(str(self.mecab_arg))
        TRAIN_FILE = "./cosme1.csv"
        file = open(TRAIN_FILE)
        line = file.readline()
        while(line):
            row = line.split(",")
            sentence = row[0]
            tags = row[1:]
            tag_list = []
            if isinstance(sentence, str):
                sentence_formatted = re.sub(self.sub_format, "", sentence)
                sentence_parsed = tagger.parse(sentence_formatted).split()
                self.parsed_sentence_length.append(len(sentence_parsed))
                input_text = sentence_parsed[:self.max_encoder_seq_length]
                self.input_texts.append(input_text)
                for word in sentence_parsed:
                    self.input_words.add(word)
                for tag in tags:
                    tag = tag.replace('\"', '').replace(',', '').rstrip()
                    if len(tag) > 0 and isinstance(tag, str):
                        tag_list.append(tag)
                        self.target_words.add(tag)
                self.target_texts.append(tag_list)
            line = file.readline()
        self.input_words = sorted(self.input_words)
        self.target_words = sorted(self.target_words)

    def prepare_for_predict(self):
        print("mecab is called")

        if not self.isTrain:
            print("error")
            return
        # Load json file and make reversed dictionary
        with open("input_indexed.json", "r") as f:
            input_words_indexed.load(f)
        with open("target_indexed.json", "r") as f:
            target_words_indexed.load(f)
        self.reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        self.reverse_target_word_index = dict((i, char) for char, i in target_token_index.items())

        # Parse input sentences by mecab
        tagger = MeCab.Tagger(str(self.mecab_arg))
        TEST_PATH = "./cosme2.csv"
        self.TEST_FILE = open(TEST_PATH)
        line = self.TEST_FILE.readline()
        while(line):
            row = line.split(",")
            sentence = row[0].replace(" ", "")
            sentence_formatted = re.sub(self.sub_format, "", content)
            sentence_parsed = tagger.parse(sentence_formatted).split()[:self.max_encoder_seq_length] # type(sentence_parsed) = list
            self.input_texts.append(sentence_parsed)
        # make encoder data
        self.encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length,), dtype='float32')
        for i, input_text in enumerate(self.input_texts):
            for j, word in enumerate(input_text):
                if input_words_indexed.get(word, 0) > 0:
                    w = input_words_indexed[word]
                else:
                    w = ukw_input_index
                self.encoder_input_data[i, j] = int(w)

    def prepare_for_training(self):
        print("prepararation is called")
        if not self.isTrain:
            print("error")
            return

        self.input_words.append('UKW')
        self.target_words.append('UKW')
        self.num_encoder_tokens = len(self.input_words) + 1
        self.num_decoder_tokens = len(self.target_words) + 1
        input_words_indexed = dict([(word, i) for i, word in enumerate(self.input_words, start=1)])
        target_words_indexed = dict([(word, i) for i, word in enumerate(self.target_words, start=1)])
        input_words_indexed['PAD'] = 0
        target_words_indexed['PAD'] = 0
        '''
        with file_io.FileIO(os.path.join(self.output_path, 'input_indexed.json'), 'w') as fo_input:
            json.dump(input_words_indexed, fo_input)
        with file_io.FileIO(os.path.join(self.output_path, 'target_indexed.json'), 'w') as fo_target:
            json.dump(target_words_indexed, fo_target)
        '''
        # 学習に使用するencoderとdecoderのデータを作成
        self.input_sequence = np.zeros((len(self.input_texts), self.max_encoder_seq_length), dtype='int32')
        self.target_sequence = np.zeros((len(self.target_texts), self.num_decoder_tokens), dtype='int32')
        ukw_input_index = input_words_indexed['UKW']
        ukw_target_index = target_words_indexed['UKW']
        # Multi-hot vectorを作成
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for j, word in enumerate(input_text):
                if input_words_indexed.get(word, 0) > 0:
                    w = input_words_indexed[word]
                else:
                    w = ukw_input_index
                self.input_sequence[i, j] = int(w)
            for k, tag in enumerate(target_text):
                if target_words_indexed.get(word, 0) > 0:
                    w = target_words_indexed[word]
                else:
                    w = ukw_target_index
                self.target_sequence[i, k] = 1

    def rnn_model(self):
        print("model is called")

        input_tensor = Input(shape=(self.max_encoder_seq_length,))
        inputs2 = Embedding(self.num_encoder_tokens, 300)(input_tensor)
        outputs1, state_h1, state_c1 = LSTM(self.latent_dim, return_sequences=True, return_state=True)(inputs2)
        #states1 = [state_h1, state_c1]
        outputs2, state_h2, state_c2 = LSTM(self.latent_dim2, return_sequences=False, return_state=True)(outputs1)
        #states2 = [state_h2, state_c2]
        dense1 = Dense(units=2000, activation="relu")(outputs2)
        dense2 = Dense(units=self.num_decoder_tokens)(dense1)
        target_tensor = Activation('softmax')(dense2)
        self.model = Model(input_tensor, target_tensor)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def train(self, is_transfer):
        self.mecab_for_train()
        self.prepare_for_training()
        self.rnn_model()
        print("train is called")
        if not self.isTrain:
            print("error")
            return
        if is_transfer:
            with file_io.FileIO(os.path.join(self.model_path, 'checkpoint.hdf5'), 'r') as reader:
                with file_io.FileIO('checkpoint.hdf5', 'w+') as writer:
                    writer.write(reader.read())
            self.model.load_weights('checkpoint.hdf5', by_name=True)
        checkpoint = keras.callbacks.ModelCheckpoint('checkpoint_att_data_kirei_all.hdf5')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=5, verbose=0, mode='auto')
        reduce_lro = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        callbacks = [checkpoint, early_stop, reduce_lro]
        self.model.fit(self.input_sequence, self.target_sequence, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks, validation_split=0.2)
        self.model.summary()
        self.model.save('tags_multihot.hdf5')
        with file_io.FileIO('tags_multihot.hdf5', 'r') as reader:
            with file_io.FileIO(os.path.join(self.model_path, 'tags_multihot.hdf5'), 'w+') as writer:
                writer.write(reader.read())

    def predict(self):
        self.prepare_for_predict()
        self.rnn_model()

        if self.isTrain:
            print("error")
            return
        # Load weights
        with file_io.FileIO(os.path.join(self.path_cloud, 'lstm_tag.hdf5'), 'r') as reader:
            with file_io.FileIO('lstm_tag.hdf5', 'w+') as writer:
                writer.write(reader.read())
        self.model.load_weights('tags_multihot.hdf5', by_name=True)
        # predict and save to the file
        f = file_io.FileIO(os.path.join(self.path_output, self.type + '.tsv'), 'w')
        f = file_io.FileIO(os.path.join(self.path_output, "sample.tsv"), "w")
        for index_sentence in range(len(self.input_texts)):
            input_sentence = self.encoder_input_data[index_sentence: index_sentence + 1]
            decoded_sentence = decode_sequence(input_sentence)
            for i in range(len(decoded_sentence)):
                f.write(str(self.TEST_FILE.iloc[index_sentence, 0]) + '\t') #TODO: change the output location
                for j in range(len(decoded_sentence[i])):
                    f.write(decoded_sentence[i][j])
                f.write('\t' + str(i+1) + '\n')
        f.close()

        def decode_sequence(input_seq):
            predict_decode_input = np.zeros((1, self.max_decoder_seq_length), dtype=np.int)
            predict_decode_input[0, 0] = self.target_token_index['\t']
            predict_out = np.zeros((10, self.max_decoder_seq_length))
            predicted_tag = self.model.predict([input_seq, predict_decode_input])
            top_n = predicted_tag[0, 0, :].argsort()[-self.num_tags - 1:]
            for i in range(self.num_tags):
                if self.reverse_target_word_index[top_n[i]] == 'EOS' or top_n[i] == 0:
                    top_n[i] = predicted_tag[0, 0, self.num_tags].argsort()[-self.num_tags - 1:]
            for i in range(self.num_tags):
                predict_decode_input[0, 1] = top_n[i]
                for j in range(1, 9):
                    predicted_tag = self.model.predict([input_seq, predict_decode_input])
                    tops = np.argsort(predicted_tag[0, j])[::-1]
                    top = np.argmax(predicted_tag[0, j, :])
                    if self.reverse_target_word_index[top] == 'EOS' or self.reverse_target_word_index[top] == 'PAD':
                        top = tops[1]
                    if self.reverse_target_word_index[top] == 'EOS' or self.reverse_target_word_index[top] == 'PAD':
                        top = tops[2]
                    if top in  predict_decode_input[0]:
                        j = 8
                    if j == 8:
                        predict_out[i] = predict_decode_input[0]
                    else:
                        predict_decode_input[0, j + 1] = top
            # one-hotの予測結果から、単語の結果を取り出す
            words = [[] for i in range(self.num_tags)]
            for i in range(self.num_tags):
                for j in range(self.num_words):
                    words[i].append(self.reverse_target_word_index[int(predict_out[i, j])])
            return words

'''メインメソッド'''
main = Tags_multihot(True)
main.train(False)
#main = Tags_multihot(False)
#main.predict()
