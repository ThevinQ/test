######################
# Vector Space Model #
######################

import numpy as np
from sklearn import datasets, feature_extraction

news_article = datasets.fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes')).data

count_vectorizer = feature_extraction.text.CountVectorizer()
x1 = count_vectorizer.fit_transform(news_article)
vocab = count_vectorizer.get_feature_names()

for i in range(4):
    ind = x1.indices[x1.indptr[i]:x1.indptr[i+1]]
    print('----' * 6)
    print('words:', [vocab[j] for j in ind])
    print('frequencies:', x1.data[x1.indptr[i]:x1.indptr[i+1]])
    print('----' * 6)
    print(news_article[i])

word_count = np.array(np.sum(x1,axis=0))
count_desc = np.sort(word_count[0,:])[::-1]

import matplotlib.pyplot as plt

plt.xscale('log')
plt.yscale('log')
plt.plot(count_desc)
plt.show()

# can also plot based on counts of frequencies, i.e., the number of words with the same frequencies

freq_count = np.bincount(word_count.flatten())
plt.xscale('log')
plt.yscale('log')
plt.plot(freq_count)
plt.show()

count_vectorizer = feature_extraction.text.CountVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x1 = count_vectorizer.fit_transform(news_article)

tfidf_transformer = feature_extraction.text.TfidfTransformer()
x2 = tfidf_transformer.fit_transform(x1)

tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x3 = tfidf_vectorizer.fit_transform(news_article)

# x2 should be the same as x3
np.linalg.norm((x2 - x3).toarray())
# or: scipy.sparse.linalg.norm(x2-x3)

# computed idf value should be same too
idf2 = tfidf_transformer.idf_
idf3 = tfidf_vectorizer.idf_

np.linalg.norm(idf2 - idf3)

################
# TruncatedSVD #
################

import numpy as np
from sklearn import datasets, feature_extraction, decomposition

news_article = datasets.fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes')).data

count_vectorizer = feature_extraction.text.CountVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x = count_vectorizer.fit_transform(news_article)
vocab = count_vectorizer.get_feature_names()

n, m = x.shape
k = 10
tSVD = decomposition.TruncatedSVD(n_components = k, random_state = 2018)
xtr = tSVD.fit_transform(x)
Sigma = np.diag([np.linalg.norm(xtr[:,i]) for i in range(k)])
U = np.zeros(xtr.shape)
for i in range(k):
    U[:,i] = xtr[:,i] / Sigma[i,i]

# Alternatively,
# Sigma_inv = np.diag([1.0 / np.linalg.norm(xtr[:,i]) for i in range(k)])
# U1 = np.matmul(xtr, Sigma_inv)
# assert: np.linalg.norm(U-U1) should be 0

V = tSVD.components_.transpose()

# Check U and V are column-orthonormal
UTU = np.matmul(U.transpose(), U)
VTV = np.matmul(V.transpose(), V)
np.linalg.norm(UTU - np.identity(k))
np.linalg.norm(VTV - np.identity(k))

# Find similar pair of words

from sklearn import metrics

sim = metrics.pairwise.cosine_similarity(V)

for i in range(m):
    for j in range(0, i+1):
        sim[i,j] = 0

for i in sim.argsort(axis = None)[::-1][:10]:
    print(sim[i // m][i % m], vocab[i // m], vocab[i % m])

####################################################################
# Alternatively, you can do manual calculation as follows -- start #
####################################################################

def cos_sim (a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

sim_i_j = []
for i in range(m):
    for j in range(i+1, m):
        sim_i_j.append((cos_sim(V[i,:],V[j,:]), i, j))

sim_sorted = sorted(sim_i_j,reverse = True)
for i in range(10):
    print(sim_sorted[i][0], vocab[sim_sorted[i][1]], vocab[sim_sorted[i][2]])

#######
# end #
#######

###############################
# Latent Dirichlet Allocation #
###############################

import numpy as np
from sklearn import datasets, feature_extraction, decomposition

news_article = datasets.fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 2018, remove = ('headers', 'footers', 'quotes')).data

count_vectorizer = feature_extraction.text.CountVectorizer(min_df = 0.01, max_df = 0.5, stop_words = 'english')
x = count_vectorizer.fit_transform(news_article)
vocab = count_vectorizer.get_feature_names()

n, m = x.shape
k = 10
lda = decomposition.LatentDirichletAllocation(n_components = k, random_state = 2018)

xtr = lda.fit_transform(x)

# Find top words for each topic
topic_word = lda.components_
for j in range(m):
    topic_word[:, j] /= sum(topic_word[:,j])

for i in range(k):
    prob_j = []
    for j in range(m):
        prob_j.append((topic_word[i,j], j))
    
    prob_sorted = sorted(prob_j,reverse = True)
    print('topic', i+1)
    for j in range(10):
        print('   ', prob_sorted[j][0], vocab[prob_sorted[j][1]])
