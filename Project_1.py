
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups
# Refer to the offcial document of scikit-learn for detailed usages:
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
categories = ['comp.graphics', 'comp.sys.mac.hardware']
# The
twenty_train = fetch_20newsgroups(subset='train', # choose which subset of the dataset to use; can be 'train', 'test', 'all'
                                  categories=None, # choose the categories to load; if is `None`, load all categories
                                  shuffle=True,
                                  random_state=42, # set the seed of random number generator when shuffling to make the outcome repeatable across different runs
#                                   remove=['headers'],
                                 )
twenty_test = fetch_20newsgroups(subset='test', categories=None, shuffle=True, random_state=42)

from pprint import pprint
pprint(list(twenty_train.target_names))
print(len(twenty_train.target))
print(twenty_train.target[8])
print(twenty_train.target[1423])


#****************** Question 1 ****************************************************************************8

plt.hist(twenty_train.target, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
plt.show();


#****************** Question 2 **************************************************************************
import numpy as np
np.random.seed(42)
import random
random.seed(42)

categories = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories,shuffle = True, random_state = None)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories,shuffle = True, random_state = None)

from sklearn.feature_extraction import text

#token pattern is regex to only allow certain words through [ref. https://regexr.com/]
stop_words = text.ENGLISH_STOP_WORDS
vectorizer = text.CountVectorizer(min_df=3, stop_words ='english', token_pattern = '[A-Za-z]\w+')

vectorizer.fit(train_dataset.data)
X_train = vectorizer.transform(train_dataset.data)
X_test = vectorizer.transform(train_dataset.data)

#verifies numbers are removed
#print(vectorizer.get_feature_names()[1:102])

import nltk
#nltk.download('punkt')#, if you need "tokenizers/punkt/english.pickle", choose it
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
wnl = nltk.wordnet.WordNetLemmatizer()

#creating Parts of Speech to hand to lemmatizer
LemmaInput = nltk.pos_tag(vectorizer.get_feature_names())
#print(LemmaInput)

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

LemmaOutput = []
for i in range(0,len(LemmaInput)):
    #pass Lemmatizer word and converted part of speech, append to list
    LemmaOutput.append(wnl.lemmatize(LemmaInput[i][0].lower(),pos=penn2morphy(LemmaInput[i][1])))

#print(len(LemmaOutput))
#vectorizer.fit(LemmaOutput)
#X_train = vectorizer.transform(train_dataset.data)
#X_test = vectorizer.transform(train_dataset.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train)
print(X_train_tfidf.shape)

X_test_tfidf = tfidf_transformer.transform(X_test)
print(X_test_tfidf.shape)


#****************** Question 3  **************************************************************************
from sklearn import decomposition

#LSI/PCA reduction method
svd = decomposition.TruncatedSVD(n_components=50, random_state=0)
X_train_reduced = svd.fit_transform(X_train_tfidf)
print(X_train_reduced.shape)

#NMF reduction method
model = decomposition.NMF(n_components=50, init='random', random_state=0)
W_train = model.fit_transform(X_train_tfidf)
print(W_train.shape)