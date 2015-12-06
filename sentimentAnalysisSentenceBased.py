# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:04:00 2015

@author: DAN
"""

from __future__ import division  # Needed in Python 2 for tokenizing.
import nltk as nltk
import dataLoadModule as dl
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

#Sentiment analysis example.
#Import a niave bayes classifier that is built into nltk.
from nltk.classify import NaiveBayesClassifier

#Import a support vector machine classifier from scikit learn.
#Use svm with a radial basis kernel function. 
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import svm

#Import the movie_reviews sentiment polarity corpus for training.
from nltk.corpus import movie_reviews
#Import a sentiment analyzer to analyze documents.
from nltk.sentiment import SentimentAnalyzer
#Import the sentiment toolkit for email analysis.
from nltk.sentiment.util import *

#Tokenizer for tokenizing the emails.
from nltk import word_tokenize


#GLOBALS
data = None

def main():
    global data
    
    #Get all of the emails.
    #data = dl.getFullEmailData()
    
 
def nltkSentimentAnalysisFirstClassifier():
    #Get all subjective and objective sentences.
    #Note: The "encode/decode" statement is used to parse the unicode representation of the text to an 
    #Ascii representation. The "apply_features()" method throws an error if this isn't done. This is most 
    #likely because python 3 uses unicode characters to perform operations on string, while python 2 doesn't.
    positive_docs = [ ([string.encode('ascii', 'ignore').decode('ascii') for string in sent], 'pos') for sent in movie_reviews.sents(categories='pos')]
    negative_docs = [ ([string.encode('ascii', 'ignore').decode('ascii') for string in sent ], 'neg') for sent in movie_reviews.sents(categories='neg')]     
    #obj_docs = [(sent.encode('ascii', 'ignore').decode('ascii'), 'obj') for sent in subjectivity.sents(categories='obj')]    
        
    #Randomly split data sets into train and test sets.
    train_pos, test_pos = train_test_split(positive_docs, test_size=1000, train_size=4000)
    train_neg, test_neg = train_test_split(negative_docs, test_size=1000, train_size=4000)
        
    #Aggregate train and test data sets.
    train = train_pos + train_neg
    test = test_pos + test_neg


    #Create a sentiment analyzer to analyze the text documents. This analyzer
    #provides an abstraction for managing a classifier, and feature extractor.
    #It also provides convinence data metrics on classifier performance. 
    sentim_analyzer = SentimentAnalyzer()
    #Mark negations in the tokenized training text, and count all negative words.
    #all_words() returns all tokens from the document, which is used to create a set 
    #of features with a feature extractor.
    all_words_with_neg_tags = sentim_analyzer.all_words([mark_negation(doc) for doc in train])
    #Create the unigram features, only taking features that occur more than 4 time.
    unigram_features = sentim_analyzer.unigram_word_feats(all_words_with_neg_tags, min_freq=2)
    #Create a feature extractor based on the unigram word features created.
    #The unigram feature extractor is found in the sentiment utils package.
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
    #Create feature-value representations of the data.
    train_set = sentim_analyzer.apply_features(train)
    test_set = sentim_analyzer.apply_features(test)
    #Note, training may take a long time.
    #Create a trainer and train the sentiment analyzer on the training set.  
    print("Begining the classifier training...")
    
    #NAIVE BAYES CLASSIFIER.
    trainer = NaiveBayesClassifier.train
    #classifier = sentim_analyzer.train(trainer, train_set)
    
    #RBF SUPPORT VECTOR MACHINE CLASSIFIER.
    clf = SklearnClassifier(svm.SVC(kernel='rbf'))
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    
    
    print("Classifier Trained! Classifying examples...")
    #Test the classifier on the test set.
    for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        print('{0}: {1}'.format(key, value))

nltkSentimentAnalysisFirstClassifier()

'''
Pickle the classifier model into a re-useable file.

SVM is fileName="svm"
NaiveBayes is fileName="nb"
DecitionTrees is fileName="dt"

'''
def saveModel(classifier, fileName):
    pickle_file = './models/' + fileName + '.pkl'
    joblib.dump(classifier, pickle_file)
    
'''
Re-load a pickle'd classifier model from a file, and return it.
'''
def loadModel(fileName):
    pickle_file = './models/' + fileName + '.pkl'   
    classifier = joblibpickle.load(pickle_file)    
    return classifier

#if __name__ == "__main__":
#    main()




