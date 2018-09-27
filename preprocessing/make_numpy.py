import re
import MeCab
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

def make_numpy():
    # set the variables
    model = Word2Vec.load("txt8corpus.model")
    df_csv = pd.read_csv("cosme1.csv", low_memory=False, header=None, dtype="str")
    df_tags = df_csv.iloc[:, 1:] # 11742
    list_tags = []
    list_vector = []
    list_sentences_1 = df_csv.iloc[:, 0].tolist() # 11742
    list_sentences_2 = []
    i = 0
    for i, sentence in enumerate(list_sentences_1):
        list_sentences_1[i] = re.sub(",", "", str(sentence))

    # make the lists of sentences and tags
    for i, sentence in enumerate(list_sentences_1):
        for j in range(df_tags.shape[1]):
            a = df_tags.iat[i, j]
            if pd.isnull(a):
                break
            list_sentences_2.append(sentence)
            list_tags.append(a)

    #ベクトルのリスト作成
    tagger = MeCab.Tagger("-Owakati")
    list_vector = []
    error = 0
    j = 0
    for i, tag in enumerate(list_tags):
        parsed_tag = tagger.parse(tag).split()
        try:
            vector = np.zeros(100,)
            if len(parsed_tag) == 1:
                vector = model.wv[tag].tolist()
                list_vector.append(vector)
            else:
                for word in parsed_tag:
                    vector += model.wv[word]
                vector = vector.tolist()
                list_vector.append(vector)
        except KeyError:
            list_tags[i] = "&" + tag
            list_vector.append(list())
            error+=1
            pass

    #　文章中にないタグのデータの削除
    delete_list = []
    for i, tag in enumerate(list_tags):
        if tag[0] == "&":
            delete_list.append(i)

    np_sentences = np.array(list_sentences_2)
    np_sentences = np.delete(np_sentences, delete_list)
    np_tags = np.array(list_tags)
    np_tags = np.delete(np_tags, delete_list)
    np_vectors = np.array(list_vector)
    np_vectors = np.delete(np_vectors, delete_list)

    return np_sentences, np_tags, np_vectors
