import re
import MeCab
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# visualize the vectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
%matplotlib inline

# set the variables
model = Word2Vec.load("txt8corpus.model")
df_csv = pd.read_csv("cosme1.csv", low_memory=False, header=None, dtype="str")
df_tags = df_csv.iloc[:, 1:]

# make unique set of tags
set_tags = set()
list_vector = []
for i in range(df_tags.shape[0]):
    for j in range(df_tags.shape[1]):
        a = df_tags.iat[i, j]
        if pd.isnull(a):
            break
        set_tags.add(a)
list_tags = list(set_tags)

# delete a word not in dictionary
tagger = MeCab.Tagger("-Owakati")
delete_list = []
for i, tag in enumerate(list_tags):
    parsed_tag = tagger.parse(tag).split()
    try:
        for word in parsed_tag:
            model.wv[word]
    except KeyError:
        delete_list.append(tag)
list_tags = [x for x in list_tags if x not in delete_list]

# make a list of vectors
for i, tag in enumerate(list_tags):
    parsed_tag = tagger.parse(tag).split()
    vector = np.zeros(100, )
    for word in parsed_tag:
        vector += model.wv[word]
    vector = vector.tolist()
    list_vector.append(vector)

# Plot the distribution of vectors
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = list_tags
    tokens = list_vector
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
#Visualize the vectors
tsne_plot(model)
