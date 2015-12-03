# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:04:00 2015

@author: DAN
"""

from __future__ import division  # Needed in Python 2 for tokenizing.
import nltk as nltk
import dataLoadModule as dl
from sklearn.cross_validation import train_test_split
import gc

#Sentiment analysis example.
#Import a niave bayes classifier.
from nltk.classify import NaiveBayesClassifier
#Import the subjectivity corpus for training.
from nltk.corpus import subjectivity
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
    
 
#def nltkSentimentAnalysisTutorial():
#Get all subjective and objective sentences.
#Note: The "encode/decode" statement is used to parse the unicode representation of the text to an 
#Ascii representation. The "apply_features()" method throws an error if this isn't done. This is most 
#likely because python 3 uses unicode characters to perform operations on string, while python 2 doesn't.
subj_docs = [ [string.encode('ascii', 'ignore').decode('ascii') for string in sent] for sent in subjectivity.sents(categories='subj')]
obj_docs = [ [string.encode('ascii', 'ignore').decode('ascii') for string in sent ] for sent in subjectivity.sents(categories='obj')]     
#obj_docs = [(sent.encode('ascii', 'ignore').decode('ascii'), 'obj') for sent in subjectivity.sents(categories='obj')]    
    
#Randomly split data sets into train and test sets.
train_subj, test_subj = train_test_split(subj_docs, test_size=100, train_size=200)
train_obj, test_obj = train_test_split(obj_docs, test_size=100, train_size=200)
    
#Aggregate train and test data sets.
train = train_subj + train_obj
test = test_subj + test_obj

#Create a sentiment analyzer to analyze the text documents. This analyzer 
#provides an abstraction for managing a classifier, and feature extractor.
#It also provides convinence data metrics on classifier performance. 
sentim_analyzer = SentimentAnalyzer()
#Mark negations in the tokenized training text, and count all negative words.
#all_words() returns all tokens from the document, which is used to create a set 
#of features with a feature extractor.
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in train])
#Create the unigram features, only taking features that occur more than 4 time.
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
#Create a feature extractor based on the unigram word features created.
#The unigram feature extractor is found in the sentiment utils package.
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
#Create feature-value representations of the data.
train_set = sentim_analyzer.apply_features(train)
test_set = sentim_analyzer.apply_features(test)
#Collect unused memory before training.
#Note, training may take a long time.
gc.collect()
#Create a trainer and train the sentiment analyzer on the training set.  
print("Begining the classifier training...")
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, train_set)
print("Classifier Trained!")
#Test the classifier on the test set.
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))

#if __name__ == "__main__":
#    main()




