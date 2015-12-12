# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 14:46:51 2015

@author: DAN
"""

import numpy as np
import pandas as pd
import sqlite3 as sql
import os

#Import the libraries for parsing the email data into the same format as that 
#used to train the classifier.
from nltk import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


data = None

'''
Returns a pandas data frame with all of the email body text.
'''
def getNonEmptyEmailBodys():
    db = sql.connect("./data/database.sqlite")      
    cursor = db.cursor()

    #Get the keys for the pandas data frame.
    cursor.execute('''PRAGMA table_info('Emails');''')
    columnNames = []
    for row in cursor:
        columnNames.append(row[1])
    
    #Retrieve all email data.
    cursor.execute('''SELECT ExtractedBodyText FROM Emails WHERE (ExtractedBodyText != '');''')
    dataDict = { 'ExtractedBodyText': []}  
    for row in cursor:
        #Extract each column value and add it to the dictionary.
        dataDict['ExtractedBodyText'].append(row[0])

    #Return pandas data frame with all data.
    data = pd.DataFrame(dataDict)
    db.close();    
    return data
    
    
'''
Returns a pandas data frame with all of the email body text tokenized. This will take 
a few seconds to run for 6741 emails with non-empty body text. This method breaks emails first 
into sentences. Each sentence is then split into tokens. Then, each token between negations is 
marked with the "_NEG" tag. This increases the accuracy of the classifier since negations are 
very important to analyzing text. Finally, each email is stored as a list of tokenized sentences. 
The resulting pandas DataFrame is returned with each row representing an email, and each entry in the column 
representing a list of tokenized sentences from the email body. 

NOTE!!!!!: This data hasn't been re-formatted in terms of the features used for sentiment analysis! This may
cause errors since the data isn't formatted in the same way. (Aka, if using unigram features, the document is represented
as a bag of words with a field representing True or False.)
'''
def getNonEmptyEmailBodysTokenized():
    db = sql.connect("./data/database.sqlite")      
    cursor = db.cursor()
    #Used to mark all words that come between negations.
    sentiment_analyzer = SentimentAnalyzer()
    #Load the pre-trained sentence tokenizer.
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')    

    #Get the keys for the pandas data frame.
    cursor.execute('''PRAGMA table_info('Emails');''')
    columnNames = []
    for row in cursor:
        columnNames.append(row[1])
    
    #Retrieve all email data.
    cursor.execute('''SELECT ExtractedBodyText FROM Emails WHERE (ExtractedBodyText != '');''')
    dataDict = { 'ExtractedBodyText': []}  
    for row in cursor:
        #Extract each column value and add it to the dictionary.
        tokenizedToSentences = sent_detector.tokenize(row[0].strip())
        tokenizedToWordsList = []
        for sentence in tokenizedToSentences:
            #Split sentences into words.
            tokenizedWords = word_tokenize(sentence)
            #Add negation tag to each word that has been negated, from the sentiment utils package.
            tokenizedWordsWithNeg = mark_negation(tokenizedWords)
            #Re-encode each word
            tokenizedWordsWithNeg = [string.encode('ascii', 'ignore').decode('ascii') for string in tokenizedWordsWithNeg]            
            #Add newly tokenized word to list.
            tokenizedToWordsList.append(tokenizedWordsWithNeg)
        #Add list of tokenized sentences to dataDict.    
        dataDict['ExtractedBodyText'].append(tokenizedToWordsList)

    #Return pandas data frame with all data.
    data = pd.DataFrame(dataDict)
    db.close()
    return data    
    
    
'''
Returns a pandas dataframe with all of the email content.
'''
def getFullEmailData():
    db = sql.connect("./data/database.sqlite")
    cursor = db.cursor()
    
    #Get the keys for the pandas data frame.
    cursor.execute('''PRAGMA table_info('Emails');''')
    columnNames = []
    for row in cursor:
        columnNames.append(row[1])
    
    #Retrieve all email data.
    dataDict = { key: [] for key in columnNames }  
    cursor.execute('''SELECT * FROM Emails WHERE (ExtractedBodyText != '');''')
    for row in cursor:
        for key in range(0, len(columnNames)):
            #Extract each column value and add it to the dictionary.
            dataDict[columnNames[key]].append(row[key])

    #Return pandas data frame with all data.
    data = pd.DataFrame(dataDict)
    db.close();
    return data
    
if __name__ == "__main__":
    data = getNonEmptyEmailBodysTokenized()
        
    




