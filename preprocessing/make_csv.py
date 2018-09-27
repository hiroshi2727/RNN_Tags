import re
import MeCab
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

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
        vector = np.zeros(100, )
        for word in parsed_tag:
            vector += model.wv[word]
        vector = vector.tolist()
        list_vector.append(vector)
    except KeyError:
        list_tags[i] = "&" + tag
        list_vector.append("a")
        pass

# list of tags to delete
delete_list = []
for i, tag in enumerate(list_tags):
    if tag[0] == "&":
        delete_list.append(i)

# form the appropriate lists of tags
np_vector = np.array(list_vector)
np_vector = np.delete(np_vector, delete_list)
np_tag = np.array(list_tags)
np_tag = np.delete(np_tag, delete_list)
list_vector = np_vector.tolist()
list_tag = np_tag.tolist()
len(list_tag)
len(list_vector)
vectors = np.zeros((np_vector.shape[0], 100))
for i, list in enumerate(list_vector):
    for j, float in enumerate(list):
        vectors[i, j] = float
