# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:04:00 2015

@author: DAN
"""

import nltk as nltk
import dataLoadModule as dl


#GLOBALS
data = None

def main():
    global data
    
    #Get all of the emails.
    data = dl.getFullEmailData()
    




if __name__ == "__main__":
    main()




