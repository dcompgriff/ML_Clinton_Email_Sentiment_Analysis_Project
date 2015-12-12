# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:04:00 2015


Todo:
DONE: 1) Run all classifiers, save, and save metrics.
DONE: 2) Make a table with all metrics, using scikit learn libraries.
DONE 3) Make set of all training documents classified by each method.
4) Make a ROC curve with all classifiers.
DONE 5) Do correlation analysis for each classifier's list of documents classified, sort, 
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
import copy as copy
import matplotlib.pyplot as plt

#ROC and auc curve metrics.
from sklearn.metrics import roc_curve, auc

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
#Import the Linear svm classifier.
from sklearn.svm import LinearSVC

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
correlationList = []
classifierNamesList = ["svm", "lr", "nb",  "dt", "mnb", "rf", "ab", "sgd"]
classifierResultsDict = {key: [] for key in classifierNamesList}
processes = []

def main():
    global data
    
    #Get all of the emails.
    #data = dl.getFullEmailData()
    #trainAllClassifiers()
    graphROCCurve()
    
'''
This method trains all of the classifiers sequentially. It first selects 4000 positive and 4000 negative examples. 
Then, it creates a bag of words set of features, with negations marked. This feature set is then applied to 
both the training set, and the test set. 

For each classifier, 4 tasks are performed:
1) The time to train is recorded.
2) The model is trained.
3) The model is saved using the "saveModel()" function.
4) The model's metrics are determined, and then stored using the "saveMetricsToFile()" method.
'''
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
    
    #SVM
    startTime = time.time()
    print("Linear Support Vector Machine.")   
    clf = SklearnClassifier(LinearSVC())
    trainer = clf.train
    classifier = sentim_analyzer.train(trainer, train_set)
    endTime = time.time()
    timeDiff = endTime - startTime
    saveModel(classifier, "lsvm")
    saveMetricsToFile("lsvm", sentim_analyzer, test_set, timeDiff/60.0)
    print "Total time to train: " + str(timeDiff/60.0) + " minutes."    
    
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
    
'''
This method is used to classify all of the emails using all classifiers. It uses the multiprocessing 
module to utilize multiple cores to run the training.
'''    
def runAllEmailClassifiersForEmails():
    global processes
  
    #Retrieve the list of emails from the data load module.
    print("Readding emails from database...")
    #Read from database. This takes a goood amount of time bc formatting and retrieving from the database.
    emailsInSentenceForm = dl.getNonEmptyEmailBodysTokenized()
    #Read from saved excel file.
    #emailsInSentenceForm = pd.read_excel("emails_to_classify.xlsx") 
    
    #Load the saved feature list.
    f = open("./bow_features.pkl", "r")   
    unigram_features = pickle.load(f)
    f.close()
   
    print("Emails read!.")      
    processes = []
    for classifierKey in classifierNamesList:
        t = Process(target=runClassifier, args=(classifierKey, emailsInSentenceForm, unigram_features, ))
        processes.append(t)
        t.start()

    print("Threads started. Waiting for all to finish.")
    for thread in processes:
        thread.join()
        
    print("All threads finished!")
     
'''
This is a worker method that is used to test each classifier. It is used with the "runAllEmailClassifiersForEmails()"
method. This method runs through all emails, and classifies each email. The resulting array of labels is saved in 
a pickled file. Every 100 emails, the pickled file is re-saved with all of the new labels. This allows for progress 
monitoring. 
'''
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
        
'''
This method compares the labels of each pair of classifiers. It calculates the correlation for each classifier, 
sorts by magnitude in descending order. It is mainly used by the "graphCorrelation()" method.
'''
def correlateClassifiers(classifierResults):
    correlationResults = []
    
    #Find all correlation values.
    for i in range(0, len(classifierNamesList) - 1):
        for j in range(i + 1, len(classifierNamesList)):
            firstClassifier = classifierNamesList[i]
            secondClassifier = classifierNamesList[j]
            correlation = np.corrcoef(classifierResults[firstClassifier], classifierResults[secondClassifier])
            correlationResults.append((firstClassifier, secondClassifier, correlation[0][1]))
        
    #Sort by highest correlation.
    sortedByHighestCorrelation = sorted(correlationResults, key=lambda item: item[2], reverse=True)
    
    return sortedByHighestCorrelation
    
'''
This method plots a comparison of all classifier correlations. 
'''
def graphCorrelation():
    global data
    global correlationList    
    
    #Get all classifier results.
    data = buildClassifierResultsTable()
    #Find all pairs of correlation. (28 total)
    correlationList = correlateClassifiers(data)
    correlationOnlyList = [item[2] for item in correlationList]    
    
    X = np.arange(0, len(correlationOnlyList))
    plt.bar(X, correlationOnlyList)
    
    for x,y in zip(X,correlationOnlyList):
        plt.text(x+0.4, y+0.05, "(" + correlationList[x][0] + ", \n" + correlationList[x][1] + ")", ha='center', va= 'bottom')
    
    plt.show()
    
'''
Used to test the classifiers on 1000 new test examples, and to return a pandas DataFrame with all classifier results,
and a DataFrame with the labels for the test set. This method is used mainly to build the data set for the ROC 
curve. When built, the results of this method are saved in excel files for quick retrieval.
'''
def classifyOn1000Examples(binary=False):
    print("Splitting positive and negative documents...")    
    positive_docs = [ ([string.encode('ascii', 'ignore').decode('ascii') for string in sent], 'pos') for sent in movie_reviews.sents(categories='pos')]
    negative_docs = [ ([string.encode('ascii', 'ignore').decode('ascii') for string in sent ], 'neg') for sent in movie_reviews.sents(categories='neg')]     
    #Randomly split data sets into train and test sets.
    train_pos, test_pos = train_test_split(positive_docs, test_size=500, train_size=4000)
    train_neg, test_neg = train_test_split(negative_docs, test_size=500, train_size=4000)
        
    #Aggregate train and test data sets.
    test = test_pos + test_neg

    #Create a sentiment analyzer to analyze the text documents. This analyzer
    #provides an abstraction for managing a classifier, and feature extractor.
    #It also provides convinence data metrics on classifier performance. 
    sentim_analyzer = SentimentAnalyzer()
    #Mark negations in the tokenized training text, and count all negative words.
    #all_words() returns all tokens from the document, which is used to create a set 
    #of features with a feature extractor.
    print("Creating feature set...")
    f = open("./bow_features.pkl", "r")   
    unigram_features = pickle.load(f)
    f.close()
    
    #Create a feature extractor based on the unigram word features created.
    #The unigram feature extractor is found in the sentiment utils package.
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
    #Create feature-value representations of the data.
    test_set = sentim_analyzer.apply_features(test)
    
    #Make a dict to hold predicted labels.
    testDict = {"test_labels": []}
    for sent in test_set:
        if binary == True:
            if sent[1] == "pos":
                testDict["test_labels"].append(1)
            else:
                testDict["test_labels"].append(-1)
        else:
            testDict["test_labels"].append(sent[1])
    
    print("Beginning classification...")
    classifierResultsDict = {key: [] for key in classifierNamesList}
    for classifierKey in classifierNamesList:
        print("Starting classifier: " + classifierKey)
        classifier = loadModel(classifierKey)
        for sent in test_set:
            label = classifier.classify(sent[0])
            if binary == True:
                if label == "pos":
                    classifierResultsDict[classifierKey].append(1)
                else:
                    classifierResultsDict[classifierKey].append(-1)
            else:
                classifierResultsDict[classifierKey].append(label)
            
    return pd.DataFrame(classifierResultsDict), pd.DataFrame(testDict)
    
'''
This method is used to graph the ROC curve for each classifier. This provides a visual method for 
analysing the classifiers.
'''
def graphROCCurve():
    #Loab initial data.
    rocDataClassified = pd.read_excel("./classifier_results/roc_data_bin.xlsx")  
    test_labels = pd.read_excel("./classifier_results/test_labels_bin.xlsx")
    
    #Calculate all metrics from the roc curve function.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for key in classifierNamesList:
        fpr[key], tpr[key], _ = roc_curve(test_labels["test_labels"], rocDataClassified[key])
        roc_auc[key] = auc(fpr[key], tpr[key])    
    
    #Plot the ROC curve for each classifier, along with the AUC measure. 
    plt.figure()
    for key in classifierNamesList:
        plt.plot(fpr[key], tpr[key], label='ROC curve of classifier {0} (area = {1:0.2f})'''.format(key, roc_auc[key]))
    
    #Set up the plot axes, and show the final figure.
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot()

'''
This method retreives the email classification results from each classifier's pickled labels file, and
creates a DataFrame holding all of the labels. This method allows for access to the email labels without 
having to re-run the classifiers. 
'''
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
    main()
    #runAllEmailClassifiersForEmails()
    #data = buildClassifierResultsTable()
    #correlationList = correlateClassifiers(data)
    #graphCorrelation()
    




