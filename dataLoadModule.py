# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 14:46:51 2015

@author: DAN
"""

import numpy as np
import pandas as pd
import sqlite3 as sql
import os


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
    cursor.execute('''SELECT * FROM Emails;''')
    for row in cursor:
        for key in range(0, len(columnNames)):
            #Extract each column value and add it to the dictionary.
            dataDict[columnNames[key]].append(row[key])

    #Return pandas data frame with all data.
    data = pd.DataFrame(dataDict)
    db.close();
    return data
    
if __name__ == "__main__":
    getNonEmptyEmailBodys()
        
    




