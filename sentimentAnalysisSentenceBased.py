# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:04:00 2015


Todo:
DONE: 1) Run all classifiers, save, and save metrics.
DONE: 2) Make a table with all metrics, using scikit learn libraries.
3) Make set of all training documents classified by each method.
4) Make a ROC curve with all classifiers.
5) Do correlation analysis for each classifier's list of documents classified, sort, 
and graph correlations from max to min.


@author: DAN
"""

from __future__ import division  # Needed in Python 2 for tokenizing.
import nltk as nltk
import time
import gc
import pickle
from multiprocessing import Process
import dataLoadModule as dl
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import pandas as pd
import numpy as np

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
classifierNamesList = ["svm", "lr", "nb",  "dt", "mnb", "rf", "ab", "sgd"]
classifierResultsDict = {key: [] for key in classifierNamesList}
processes = []

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
    
    #Save the unigram feature list to a file so it can be used later.
    #These features need to be applied to the email set.
    f = open("./bow_features.pkl", "w")   
    pickle.dump(unigram_features, f)
    f.close()
    
    #Create a feature extractor based on the unigram word features created.
    #The unigram feature extractor is found in the sentiment utils package.
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
    #Create feature-value representations of the data.
    train_set = sentim_analyzer.apply_features(train)
    test_set = sentim_analyzer.apply_features(test)
    
    #Collect some memory.
    positive_docs = None
    negative_docs = None
    gc.collect()    
    
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
    clf = SklearnClassifier(SGDClassifier())
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
    saveMetricsToFile("svm", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
    #Multinomial Naive Bayes.
    startTime = time.time()
    print("Multinomial Naive Bayes.")   
    clf = SklearnClassifier(MultinomialNB())
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
    clf = SklearnClassifier(LogisticRegression())
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
    clf = SklearnClassifier(DecisionTreeClassifier())
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
    clf = SklearnClassifier(RandomForestClassifier())
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
    clf = SklearnClassifier(AdaBoostClassifier())
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "ab")
    saveMetricsToFile("ab", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."
    
def runAllEmailClassifiersForEmails():
    global processes
    #Run all classifiers in parallel.
#    with ThreadPoolExecutor(max_workers=4) as e:
#        e.submit(shutil.copy, 'src1.txt', 'dest1.txt')
#        e.submit(shutil.copy, 'src2.txt', 'dest2.txt')
#        e.submit(shutil.copy, 'src3.txt', 'dest3.txt')
#        e.submit(shutil.copy, 'src4.txt', 'dest4.txt')    

    
    
    
    #Create a sentiment analyzer to analyze the text documents. This analyzer
    #provides an abstraction for managing a classifier, and feature extractor.
    #It also provides convinence data metrics on classifier performance. 
#    sentim_analyzer = SentimentAnalyzer()
#    #Load the saved feature list.
#    f = open("./bow_features.pkl", "r")   
#    unigram_features = pickle.load(f)
#    f.close()
#    #Create a feature extractor based on the unigram word features created.
#    #The unigram feature extractor is found in the sentiment utils package.
#    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
#    
#    #Retrieve the list of emails from the data load module.
    print("Readding emails from database...")
#    #Read from database. This takes a goood amount of time bc formatting and retrieving from the database.
    emailsInSentenceForm = dl.getNonEmptyEmailBodysTokenized()
#    #Read from saved excel file.
#    #emailsInSentenceForm = pd.read_excel("emails_to_classify.xlsx") 
    
    #Load the saved feature list.
    f = open("./bow_features.pkl", "r")   
    unigram_features = pickle.load(f)
    f.close()
#    
    print("Emails read!.")    
    
    processes = []
    for classifierKey in classifierNamesList:
        t = Process(target=runClassifier, args=(classifierKey, emailsInSentenceForm, unigram_features, ))
        processes.append(t)
        t.start()
    
    print("Threads created.")    
#    for thread in processes:
#        thread.start() 
#        
#    #Wait for all threads to finish.
    print("Threads started. Waiting for all to finish.")
    for thread in processes:
        thread.join()
#        
    print("All threads finished!")
    
    #For each classifier, parse the email data, and classify it.
#    for classifierKey in classifierNamesList:
#        print("Current classifier: " + classifierKey)
#        #Load the classifier.
#        classifier = loadModel(classifierKey)
#        
#        #Set up sentiment counts for the emails.
#        posVote = 0
#        negVote = 0
#        for emailIndex in range(0, emailsInSentenceForm.shape[0]):
#            #if(emailIndex % 100 == 0):
#            print(classifierKey + " On email: " + str(emailIndex))
#            #Get the list of sentences for the email.
#            email = emailsInSentenceForm.iloc[emailIndex,:][0]
#            featurizedSentenceList = sentim_analyzer.apply_features(email)
#            for sent in featurizedSentenceList:
#                label = classifier.classify(sent[0])
#                if label == "pos":
#                    posVote += 1
#                else:
#                    negVote += 1
#            #Take the maximum vote for the class label. Use 1 and -1 to faciliitate the later correlation calculations.
#            if posVote >= negVote:
#                classifierResultsDict[classifierKey].append(1)
#            else:
#                classifierResultsDict[classifierKey].append(-1)
#            #Reset pos and neg votes to 0.
#            posVote = 0
#            negVote = 0
#            
#        #Convert the classifier results to a pandas data frame.
#        finalData = pd.DataFrame(classifierResultsDict)
#        finalData.to_excel("email_classifier_results.xlsx")
#        return finalData
    
    
    
def runClassifier(classifierKey, emailsInSentenceForm, unigram_features):
    #Create a sentiment analyzer to analyze the text documents. This analyzer
    #provides an abstraction for managing a classifier, and feature extractor.
    #It also provides convinence data metrics on classifier performance. 
    sentim_analyzer = SentimentAnalyzer()
    #Create a feature extractor based on the unigram word features created.
    #The unigram feature extractor is found in the sentiment utils package.
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)    
    
    
    print("Current classifier: " + classifierKey)
    #Load the classifier.
    classifier = loadModel(classifierKey)
    
    #Set up sentiment counts for the emails.
    posVote = 0
    negVote = 0
    classifierResultsDict = []
    #int(emailsInSentenceForm.shape[0]/3.0)
    for emailIndex in range(0, int(emailsInSentenceForm.shape[0])):
        if(emailIndex % 100 == 0):
            print(classifierKey + " On email: " + str(emailIndex))
            #Write the results to a file.
            f = open("./"+classifierKey+"Results.pkl", "w")   
            pickle.dump(classifierResultsDict, f)
            f.close()
        #Get the list of sentences for the email.
        email = emailsInSentenceForm.iloc[emailIndex,:][0]
        featurizedSentenceList = sentim_analyzer.apply_features(email)
        for sent in featurizedSentenceList:
            label = classifier.classify(sent[0])
            if label == "pos":
                posVote += 1
            else:
                negVote += 1
        #Take the maximum vote for the class label. Use 1 and -1 to faciliitate the later correlation calculations.
        if posVote >= negVote:
            classifierResultsDict.append(1)
        else:
            classifierResultsDict.append(-1)
        #Reset pos and neg votes to 0.
        posVote = 0
        negVote = 0
        
        
    #Write the results to a file.
    f = open("./"+classifierKey+"Results.pkl", "w")   
    pickle.dump(classifierResultsDict, f)
    f.close()
        

def correlateClassifiers(classifierResults):
    #Example Dictionary.
#    exList = ["svm", "dt", "nb"]
#    cDict = {"svm": [1, 1, -1, 1, 1], "dt": [-1, 1, 1, 1, 1], "nb": [-1, 1, 1, 1, 1]}  
    #List of tupples (firstClassifierIndex, secondClassifierIndex, correlationValue)
#    correlationResults = []
#    classifierResults = pd.DataFrame(cDict)
    
    #Find all correlation values.
    for i in range(0, len(classifierNamesList) - 1):
        for j in range(i + 1, len(classifierNamesList)):
            firstClassifier = classifierNamesList[i]
            secondClassifier = classifierNamesList[j]
            correlation = np.corrcoef(classifierResults[firstClassifier], classifierResults[secondClassifier])
            correlationResults.append((i, j, correlation[0][1]))
        
    #Sort by highest correlation.
    sortedByHighestCorrelation = sorted(correlationResults, key=lambda item: item[2], reverse=True)
    
    return sortedByHighestCorrelation
    
def buildClassifierResultsTable():
    for classifierKey in classifierNamesList:
        f = open("./classifier_results/"+classifierKey+"Results.pkl", "r")   
        resultsList = pickle.load(f)
        f.close()        
        classifierResultsDict[classifierKey] = resultsList   
        
    return pd.DataFrame(classifierResultsDict)
    

'''
Pickle the classifier model into a re-useable file.

Stochastic Gradient Descent is fileName="sgd"
SVM is fileName="svm"
NaiveBayes is fileName="nb"
Multinomial NaiveBayes is fileName="mnb"
DecisionTrees is fileName="dt"
AdaBoost is fileName="ab"
Random Forrest is fileName="rf"
Logistic Regression is fileName="lr"


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
    #main()
    #runAllEmailClassifiersForEmails()
    data = buildClassifierResultsTable()
    #correlateClassifiers()
    




