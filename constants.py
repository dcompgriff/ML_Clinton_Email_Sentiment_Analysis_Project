from enum import Enum

"""
Enums
"""
class FeatureExtractor(Enum):
    bag_of_words = 0
    unigram = 1
    bigram_bag_of_words = 2
    adjective_bag_of_words = 3 # requires mark_negation to be false
    pos_bag_of_words = 4
    freq_dist = 5

class Corpus(Enum):
    # movie_review = 0
    pol_debates = 1

class Classifier(Enum):
    naive_bays = 0
    #maxent = 1
    decision_tree = 2
    linear_svm = 3
    random_forest = 4
    logistic = 5


"""
Constants for Document-Based
"""
mark_negation = False
feature_extractor = FeatureExtractor.adjective_bag_of_words
corpus = Corpus.pol_debates
classifier = Classifier.naive_bays

hunpos_english_model = "/Users/jeffnainap/Applications/bin/english.model"

"""
Constants for General Statistics
"""
country_list = "data/countries.txt"
political_figures = "data/political_figures.txt"