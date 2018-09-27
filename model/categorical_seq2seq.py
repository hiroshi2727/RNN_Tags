#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import keras
from keras import backend as K
from keras.activations import  softmax
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, Permute, Multiply, Lambda, Activation
from keras.layers.wrappers import TimeDistributed
import numpy as np
import os
import pandas as pd
import MeCab
import re
import json

class Tags_seq2seq:
    # predictionとtraining共有の変数を設定
    batch_size = 64  # Batch size for training.
    epochs = 10  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    latent_dim2 = 128
    num_samples = 120000  # Number of samples to train on.
    max_content = 300
    max_sentence_length = 300
    # リストを作成
    input_texts = list() # parsed sentences
    target_texts = list() # parsed tags
    input_chars = set() # each character of parsed sentences
    target_chars = set() # each character of parsed tags
    parsed_sentence_length = list()
    parsed_tag_length = list()

    def __init__(self, isTrain):
        '''
        #　引数
            isTrain: 作成されるオブジェクトがtraining用かpredict用か。training用の場合はtrue,　predict用の場合はfalse。
        '''
        self.isTrain = isTrain

    def set_training_variables(self):
        '''trainingモデル用の各パスを指定。
        # 引数
        '''
        if not self.isTrain:
            print("このfunctionはpredictオブジェクト用です。set_predict_variables()を使用してください。")
            return
        # training用のベースフォルダのパスを指定
        self.base_path = 'gs://cosme/tags/train/maru_att'
        self.model_path = self.base_path + 'model'
        self.output_path = self.base_path + 'char_dict'
        input_path = self.base_path + 'train_data'
        input_path_list_tf = tf.gfile.ListDirectory(input_path)
        self.input_path_list = map(lambda file: os.path.join(input_path, file), input_path_list_tf)
        self.max_tag_length = 7
        self.max_char_length = 20000
        self.max_texts = -1

    def set_predict_variables(self, project_name, type_name, input_name, output_name):
        '''predictモデル用の各パスを指定。
        # 引数
            project_name:　プロジェクトパスの名前
            type_name: タイプパスの名前
            input_name: インプットパスの名前
            output_name: アウトプットパスの名前
        '''
        if self.isTrain:
            print("このfunctionはtrainingオブジェクト用です。set_training_variables()を使用してください。")
            return
        # prediction用のパスを設定
        self.type = type_name
        self.path_tags = 'gs://' + project_name + '/tags/'
        self.path_cloud = self.path_tags + 'training/201805/model/'
        self.path_dict = self.path_tags+ 'training/201805/dict/'
        self.path_input = self.path_tags + 'input/' + input_name + '/' + type_name + '/' + type_name + '.csv'
        self.path_output = self.path_tags + 'output/' + output_name + '/' + type_name + '/'
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #TODO: 使われていない変数
        # prediction用の変数を設定
        self.input_token_index = json.loads(tf.gfile.Open(self.path_dict + 'fo_input.json').readline())
        self.target_token_index = json.loads(tf.gfile.Open(self.path_dict + 'fo_target.json').readline())
        self.num_encoder_tokens = len(self.input_token_index)
        self.num_decoder_tokens = len(self.target_token_index) + 1
        self.max_encoder_seq_length = 300
        self.max_decoder_seq_length = 9
        self.num_tags = 10
        self.num_words = 9



    def mecab_for_predict(self):
        ''' MeCabを使用した形態素解析
        # 引数
        '''
        if not self.isTrain:
            print("このfunctionはpredictオブジェクト用です。mecab_for_train()を使用してください。")
            return
        # MeCabのtaggerのセットアップ
        mecab_arg = '-F\s%f[6] -U\s%m -E\\n'
        sub_format = u"(https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+)|[!-~]|[︰-＠]|[)(＜＞＆。＿％『→；〜．「」、↓♪■］［∴・…]"
        tagger = MeCab.Tagger(str(mecab_arg))
        # tag_dataファイルの形態素解析 (predictオブジェクト)
        self.tag_data = pd.read_csv(tf.gfile.Open(self.path_input), header=None, engine='python')
        for index, row in self.tag_data.iterrows():
            try:
                content = row[1].replace(" ", "").replace("　", "")
                content_format = re.sub(sub_format, "", content)
                content_mecab = tagger.parse(content_format).split()[:self.max_content]
                self.input_texts.append(content_mecab)
            except Exception as e:
                print(e)
        self.encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length,), dtype='float32')
        for i, input_text in enumerate(self.input_texts):
            t = 0
            for char in input_text:
                decoded_string = ''
                try:
                    decoded_string = char.decode('utf-8')
                    input_index = self.input_token_index.get(decoded_string, 0)
                    if input_index > 0:
                        self.encoder_input_data[i, t] = input_index
                        t += 1
                except Exception as e:
                    print(e)
        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

        def mecab_for_train(self, test):
            ''' MeCabを使用した形態素解析
            # 引数
                test: 形態素解析後にテストをするかどうか。
            '''
            if not self.isTrain:
                print("このfunctionはtrainingオブジェクト用です。mecab_for_predict()を使用してください。")
                return
            # MeCabのtaggerのセットアップ
            mecab_arg = '-F\s%f[6] -U\s%m -E\\n'
            sub_format = u"(https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+)|[!-~]|[︰-＠]|[)(＜＞＆。＿％『→；〜．「」、↓♪■］［∴・…]"
            tagger = MeCab.Tagger(str(mecab_arg))
            # パスから各ファイルの取得
            for file in self.input_path_list:
                file_tf = tf.gfile.Open(file)
                line = file_tf.readline()
                while (line): # check if the line contains EOS
                    row = line.split(',')
                    try:
                        if isinstance(row[0], str):
                            # 文章の形態素解析
                            sentence = row[0] # extract the sentence from a row in a file
                            sentence_formatted = re.sub(sub_format, "", sentence) # delete the redundant strings from the sentence
                            sentence_parsed = tagger.parse(sentence_formatted).split() # split the sentence by comma which were separated
                            self.parsed_sentence_length.append(len(sentence_parsed)) # add the length of words to the list
                            sentence_parsed = sentence_parsed[:self.max_sentence_length] # 長さ調整
                            # タグの形態素解析
                            tags = row[1:]
                            for tag in tags:
                                tag = tag.replace('\"', '').replace(',', '')
                                if len(tag) > 3 and isinstance(tag, str): #TODO if文の4以上の必要性確認
                                    tag_parsed_splitted = tagger.parse(tag).rstrip().split()
                                    self.parsed_tag_length.append(len(tag_parsed_splitted))
                                    target_text = ['\t'] + tag_parsed_splitted[:self.max_tag_length] + ['EOS']
                                    if 2 < len(target_text): #TODO tagが4文字以上のstringであると確認しているので、if分不要なはず
                                        self.input_texts.append(sentence_parsed) #TODO len(tags)のlistを作ってその回数だけtarget_textsと照応させる方がbetter?
                                        self.target_texts.append(target_text)
                                        for char in sentence_parsed:
                                            self.input_chars.add(char)
                                        for char in target_text:
                                            self.target_chars.add(char)
                        line = file_tf.readline()
                    except TypeError as e:
                        print(e)
            # sort the list
            self.input_chars = sorted(list(self.input_chars))
            self.target_chars = sorted(list(self.target_chars))
            # Mini test for data
            if test:
                self.output_path = self.base_path + 'test/char_dict'
                self.model_path = 'gs://cosme/tags/model/test'
                self.max_char_length = 1000
                self.max_texts = 1000
                self.input_chars = self.input_chars[:self.max_char_length]
                self.target_chars = self.target_chars[:self.max_char_length]
                self.input_texts = self.input_texts[:self.max_texts]
                self.target_texts = self.target_texts[:self.max_texts]

    def prepare_for_training(self):
        '''
        '''
        if not self.isTrain:
            print("このfunctionはtrainingオブジェクト用です。predictオブジェクトでは使用できません。")
            return
        self.input_chars.append('UKW')
        self.target_chars.append('UKW')
        self.num_encoder_tokens = len(self.input_chars) + 1
        self.num_decoder_tokens = len(self.target_chars) + 1
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])
        input_chars_indexed = dict([(char, i) for i, char in enumerate(self.input_chars, start=1)])
        target_chars_indexed = dict([(char, i) for i, char in enumerate(self.target_chars, start=1)])
        input_chars_indexed['PAD'] = 0
        target_chars_indexed['PAD'] = 0
        with file_io.FileIO(os.path.join(self.output_path, 'fo_input.json'), 'w') as fo_input:
            json.dump(input_chars_indexed, fo_input)
        with file_io.FileIO(os.path.join(self.output_path, 'fo_target.json'), 'w') as fo_target:
            json.dump(target_chars_indexed, fo_target)
        # 学習に使用するencoderとdecoderのデータを作成する。
        self.encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length), dtype='int32')
        self.decoder_input_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length), dtype='int32')
        self.decoder_target_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens), dtype='int32')
        ukw_input_index = input_chars_indexed['UKW'] # words not in dictionary = ukw
        ukw_target_index = target_chars_indexed['UKW']
        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                if input_chars_indexed.get(char, 0) > 0:
                    w = input_chars_indexed[char]
                else:
                    w = ukw_input_index
                self.encoder_input_data[i, t] = int(w)
            for t, char in enumerate(target_text):
                if target_chars_indexed.get(char, 0) > 0:
                    w = target_chars_indexed[char]
                else:
                    w = ukw_target_index
                self.decoder_input_data[i, t] = int(w)
                if t > 0:
                    self.decoder_target_data[i, t - 1, w] = 1

    def seq2seq_model(self):
        '''sequence to sequenceのattention modelを作成する。'''
        # encoder
        encoder_inputs = Input(shape=(self.max_encoder_seq_length,))
        encoder_inputs2 = Embedding(self.num_encoder_tokens, 300)(encoder_inputs)
        encoder_lstm1 = LSTM(self.latent_dim, return_sequences=True,return_state=True)
        encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_inputs2)
        encoder_states1 = [state_h1, state_c1]
        encoder_lstm2 = LSTM(self.latent_dim2,return_sequences=True,return_state=True)
        encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)
        encoder_states2 = [state_h2, state_c2]
        # decoder
        decoder_inputs = Input(shape=(self.max_decoder_seq_length,))
        decoder_inputs2 = Embedding(self.num_decoder_tokens, 300)(decoder_inputs)
        decoder_lstm1 = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_lstm2 = LSTM(self.latent_dim2, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm1(decoder_inputs2, initial_state=encoder_states1)
        decoder_outputs2, _2, __2 = decoder_lstm2(decoder_outputs, initial_state=encoder_states2)
        # attention model
        annotation = TimeDistributed(Dense(units=self.latent_dim2))(encoder_outputs2)
        rep_enc = TimeDistributed(RepeatVector(self.max_decoder_seq_length))
        rep_encoded = Permute((2,1,3), input_shape=(self.max_encoder_seq_length, self.max_decoder_seq_length, self.latent_dim2))(rep_enc(encoder_outputs2))
        rep_dec = TimeDistributed(RepeatVector(self.max_encoder_seq_length))
        rep_decoded = rep_dec(decoder_outputs2)
        rep_ann = Permute((2, 1, 3), input_shape=(self.max_encoder_seq_length, self.max_decoder_seq_length, self.latent_dim2))(rep_enc(annotation))
        # context
        score_elem = Multiply([rep_decoded, rep_ann])
        score = Lambda(lambda x: K.sum(x,axis=3, keepdims= True))(score_elem)
        attention_weight = Lambda(lambda x: softmax(x, axis=2))(score)
        context_elem = Multiply([rep_encoded, attention_weight])
        context = Lambda(lambda x: K.sum(x, axis=2))(context_elem)
        dec_and_att = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([decoder_outputs2, context])
        # prediction (dense = fully connected layer)
        dense1 = TimeDistributed(Dense(units=self.latent_dim2*2))(dec_and_att)
        dense1_act = Activation('tanh')(dense1)
        dense2 = TimeDistributed(Dense(units=self.num_decoder_tokens))(dense1_act)
        dense2_act = Activation('softmax')(dense2)
        #　モデルを作成する。
        self.model = Model([encoder_inputs, decoder_inputs], dense2_act)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    def train_model(self, is_transfer):
        '''モデルに学習させ、ファイルに保存する。必要であれば転移学習させる。
        # 引数
            is_tranfer: 転移学習させるかどうか。trueならさせる。
        '''
        if not self.isTrain:
            print("このfunctionはtrainingオブジェクト用です。predictオブジェクトでは使用できません。")
            return
        #　以前の重みを読み込んで転移学習させる。
        if is_transfer:
            with file_io.FileIO(os.path.join(self.model_path, 'checkpoint.hdf5'), 'r') as reader:
                with file_io.FileIO('checkpoint.hdf5', 'w+') as writer:
                    writer.write(reader.read())
            self.model.load_weights('checkpoint.hdf5', by_name=True)
        # コールバック関数を指定して学習させる。
        checkpoint = keras.callbacks.ModelCheckpoint('checkpoint_att_data_kirei_all.hdf5')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=5, verbose=0, mode='auto')
        reduce_lro = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        callbacks = [checkpoint, early_stop, reduce_lro]
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks, validation_split=0.2)
        self.model.save('checkpoint_att_data_kirei_nogpu_all.hdf5')
        with file_io.FileIO('checkpoint_att_data_kirei_nogpu_all.hdf5', 'r') as reader:
            with file_io.FileIO(os.path.join(self.model_path, 'checkpoint_att_data_kirei_nogpu_all.hdf5'), 'w+') as writer:
                writer.write(reader.read())

    def predict_model(self, is_transfer):
        '''モデルに予測させ、結果をファイルに保存させる。必要であれば転移学習させる。
        # 引数
            input_seq: 予測させる文字列。
            is_tranfer: 転移学習させるかどうか。trueならさせる。
        '''
        if self.isTrain:
            print("このfunctionはpredictオブジェクト用です。trainingオブジェクトでは使用できません。")
            return
        if is_transfer: #　過去に学習した重みを読み込んで転移学習させる。
            with file_io.FileIO(os.path.join(self.path_cloud, 'lstm_tag.hdf5'), 'r') as reader:
                with file_io.FileIO('lstm_tag.hdf5', 'w+') as writer:
                    writer.write(reader.read())
            self.model.load_weights('lstm_tag.hdf5', by_name=True)
        def decode_sequence(input_seq):
            predict_decode_input = np.zeros((1, self.max_decoder_seq_length), dtype=np.int)
            predict_decode_input[0, 0] = self.target_token_index['\t']
            predict_out = np.zeros((10, self.max_decoder_seq_length))
            predicted_tag = self.model.predict([input_seq, predict_decode_input])
            top_n = predicted_tag[0, 0, :].argsort()[-self.num_tags - 1:]
            for i in range(self.num_tags):
                if self.reverse_target_char_index[top_n[i]] == 'EOS' or top_n[i] == 0:
                    top_n[i] = predicted_tag[0, 0, self.num_tags].argsort()[-self.num_tags - 1:]
            for i in range(self.num_tags):
                predict_decode_input[0, 1] = top_n[i]
                for j in range(1, 9):
                    predicted_tag = self.model.predict([input_seq, predict_decode_input])
                    tops = np.argsort(predicted_tag[0, j])[::-1]
                    top = np.argmax(predicted_tag[0, j, :])
                    if self.reverse_target_char_index[top] == 'EOS' or self.reverse_target_char_index[top] == 'PAD':
                        top = tops[1]
                    if self.reverse_target_char_index[top] == 'EOS' or self.reverse_target_char_index[top] == 'PAD':
                        top = tops[2]
                    if top in  predict_decode_input[0]:
                        j = 8
                    if j == 8:
                        predict_out[i] = predict_decode_input[0]
                    else:
                        predict_decode_input[0, j + 1] = top
            # 予測結果をファイルに保存する。
            words = [[] for i in range(self.num_tags)]
            for i in range(self.num_tags):
                for j in range(self.num_words):
                    words[i].append(self.reverse_target_char_index[int(predict_out[i, j])])
            return words
        #　ファイルに保存する。
        f = file_io.FileIO(os.path.join(self.path_output,self.type + '.tsv'), 'w')
        f = file_io.FileIO(os.path.join(self.path_output, "sample.tsv"), "w")
        for seq_index in range(len(self.encoder_input_data)):
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(input_seq)
            for i in range(len(decoded_sentence)):
                f.write(str(self.tag_data.iloc[seq_index, 0]) + '\t')
                for j in range(len(decoded_sentence[i])):
                    f.write(decoded_sentence[i][j])
                f.write('\t' + str(i+1) + '\n')
        f.close()
