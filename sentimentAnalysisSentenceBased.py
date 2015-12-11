# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:04:00 2015


Todo:
1) Run all classifiers, save, and save metrics.
2) Make a table with all metrics, using scikit learn libraries.
3) Make set of all training documents classified by each method.
4) Make a ROC curve with all classifiers.
5) Do correlation analysis for each classifier's list of documents classified, sort, 
and graph correlations from max to min.


@author: DAN
"""

from __future__ import division  # Needed in Python 2 for tokenizing.
import nltk as nltk
import time
import dataLoadModule as dl
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


#Import the base scikit learn wrapper class from nltk.
from nltk.classify.scikitlearn import SklearnClassifier
#Import the svm classifier from scikit learn.
from sklearn import svm
#Import a niave bayes classifier that is built into nltk.
from nltk.classify import NaiveBayesClassifier
#Import the multinomial naive bayes classifier.
from sklearn.naive_bayes import MultinomialNB
#Import the logistic regression classifier.
from sklearn.linear_model import LogisticRegression
#Import the stochastic gradient descent classifier.
from sklearn.linear_model import SGDClassifier
#Import the decision tree classifier.
from sklearn.tree import DecisionTreeClassifier
#Import the random forrests classifier.
from sklearn.ensemble import RandomForestClassifier
#Import the adaboost classifier.
from sklearn.ensemble import AdaBoostClassifier

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
    #trainAllClassifiers()
    
 
def trainAllClassifiers():
    #Get all subjective and objective sentences.
    #Note: The "encode/decode" statement is used to parse the unicode representation of the text to an 
    #Ascii representation. The "apply_features()" method throws an error if this isn't done. This is most 
    #likely because python 3 uses unicode characters to perform operations on string, while python 2 doesn't.
    print("Splitting positive and negative documents...")    
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
    print("Creating feature set...")
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
    print("Beginning the classifier training...")
    
    #Naive Bayes
    startTime = time.time()
    print("Naive Bayes.")   
    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "nb")
    saveMetricsToFile("nb", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."    
    
    #Stochastic Gradient Descent. (Performed first since it takes the least amount of time.)
    startTime = time.time()
    print("Stochastic Gradient Descent.")   
    clf = SklearnClassifier(SGDClassifier)
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "sgd")
    saveMetricsToFile("sgd", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."    
    
    #SVM
    startTime = time.time()
    print("RBF Support Vector Machine.")   
    clf = SklearnClassifier(svm.SVC(kernel='rbf'))
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "svm")
    saveMetricsToFile("svm", sentim_analyzer, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
    #Multinomial Naive Bayes.
    startTime = time.time()
    print("Multinomial Naive Bayes.")   
    clf = SklearnClassifier(MultinomialNB)
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "mnb")
    saveMetricsToFile("mnb", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
    #Logistic Regression.
    startTime = time.time()
    print("Logistic Regression.")   
    clf = SklearnClassifier(LogisticRegression)
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "lr")
    saveMetricsToFile("lr", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
    #Descision tree
    startTime = time.time()
    print("Decision Tree.")   
    clf = SklearnClassifier(DecisionTreeClassifier)
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "dt")
    saveMetricsToFile("dt", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
    #Random Forrest.
    startTime = time.time()
    print("Random Forrest.")   
    clf = SklearnClassifier(RandomForestClassifier)
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "rf")
    saveMetricsToFile("rf", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
    #Adaboost
    startTime = time.time()
    print("Ada Boost")   
    clf = SklearnClassifier(AdaBoostClassifier)
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "ab")
    saveMetricsToFile("ab", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    

'''
Pickle the classifier model into a re-useable file.

Stochastic Gradient Descent is fileName="sgd"
SVM is fileName="svm"
NaiveBayes is fileName="nb"
Multinomial NaiveBayes is fileName="mnb"
DecisionTrees is fileName="dt"


'''
def saveModel(classifier, fileName):
    pickle_file = './models/' + fileName + '.pkl'
    joblib.dump(classifier, pickle_file)
    
'''
Re-load a pickle'd classifier model from a file, and return it.
'''
def loadModel(fileName):
    pickle_file = './models/' + fileName + '.pkl'   
    classifier = joblib.load(pickle_file)    
    return classifier
    
'''
Save the metrics of the classifier to a text file.
'''
def saveMetricsToFile(fileName, sentim_analyzer, test_set, timeInMin):
    f = open("./models/results/" + fileName + ".txt", "w")
    #Print time to train in minutes.
    f.write("Minutes to train: " + str(timeInMin) + "\n")    
    #Test the classifier on the test set.
    for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        f.write('{0}: {1}'.format(key, value) + "\n") 
        
    f.close()   


if __name__ == "__main__":
    main()




