from __future__ import division
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize
from collections import Counter
import codecs

import socket
socket.getaddrinfo('localhost', 8080)

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            return mode(votes)
        # data = Counter(votes)
        # # Returns the highest occurring item
        # print "\nHighest Occurring item :"+str(data.most_common(1))
        # return data.most_common(1)[0][1]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        data = Counter(votes)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        # Returns the highest occurring item
        # print "\nConfidence"+str(data.most_common(1)[0][1])
        # return data.most_common(1)[0][1]
        
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

short_pos = short_pos.decode('unicode_escape').encode('ascii','ignore')
short_neg = short_neg.decode('unicode_escape').encode('ascii','ignore')

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:      
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

##
### negative data example:      
##training_set = featuresets[100:]
##testing_set =  featuresets[:100]


save_classifier = open("twit_original_naive","rb")
classifier=pickle.load(save_classifier)
save_classifier.close()

# classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

# save_classifier = open("twit_original_naive","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

save_classifier = open("twit_mnb","rb")
MNB_classifier=pickle.load(save_classifier)
save_classifier.close()

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# save_classifier = open("twit_mnb","wb")
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()

save_classifier = open("twit_bernoulli","rb")
BernoulliNB_classifier=pickle.load(save_classifier)
save_classifier.close()

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

# save_classifier = open("twit_bernoulli","wb")
# pickle.dump(BernoulliNB_classifier, save_classifier)
# save_classifier.close()


save_classifier = open("twit_logistic","rb")
LogisticRegression_classifier=pickle.load(save_classifier)
save_classifier.close()

# LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
# LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# save_classifier = open("twit_logistic","wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()

save_classifier = open("twit_SGDC","rb")
SGDClassifier_classifier=pickle.load(save_classifier)
save_classifier.close()

# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# save_classifier = open("twit_SGDC","wb")
# pickle.dump(SGDClassifier_classifier, save_classifier)
# save_classifier.close()

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

save_classifier = open("twit_linearsvc","rb")
LinearSVC_classifier=pickle.load(save_classifier)
save_classifier.close()

# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# save_classifier = open("twit_linearsvc","wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()

save_classifier = open("twit_nusvc","rb")
NuSVC_classifier=pickle.load(save_classifier)
save_classifier.close()

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# save_classifier = open("twit_nusvc","wb")
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()


voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)