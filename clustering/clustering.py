import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
%matplotlib inline

# エルボー法で最適なクラスタの数を求める
distortions = []
X = list_vector
for i  in range(130, 200):
    print("iteration", i)
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(130, 200), distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# k-means clustering
vectors = np.zeros((np_vector.shape[0], 100))
for i, list in enumerate(list_vector):
    for j, float in enumerate(list):
        vectors[i, j] = float
n_clusters = 100
model = KMeans(n_clusters=n_clusters, max_iter=1000).fit(vectors)
labels = model.labels_
label_list = [[] for i in range(n_clusters)]
for i, list in enumerate(label_list):
    for j, label in enumerate(labels):
        if i == label:
            label_list[i].append(list_tag[j])

# second-time clustering
model_wv = Word2Vec.load("txt8corpus.model")
n2 = 40
make_vector = []
for word in b:
    try:
        make_vector.append(model_wv.wv[word])
    except KeyError:
        pass
model2 = KMeans(n_clusters=n2).fit(make_vector)
labels2 = model2.labels_
label_list2 = [[] for i in range(n2)]
for i, list in enumerate(label_list2):
    for j, label in enumerate(labels2):
        if i == label:
            label_list2[i].append(b[j])
print(label_list2)

# Save results to csv file
concat_list = []
for i, list in enumerate(label_list):
    label_list[i] = pd.Series(label_list[i]).T
    concat_list.append(label_list[i])
concat = pd.concat(concat_list, axis=1)
concat.shape
concat.to_csv("vectors_clustering2.csv")
