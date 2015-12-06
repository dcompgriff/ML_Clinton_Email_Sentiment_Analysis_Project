from enum import Enum

"""
Enums
"""
class FeatureExtractor(Enum):
    bag_of_words = 0
    unigram = 1
    big_gram = 2
    adjective_bag_of_words = 3 # requires mark_negation to be false, doesn't yet work...

class Corpus(Enum):
    movie_review = 0

class Classifier(Enum):
    naive_bays = 0


"""
Constants
"""
mark_negation = False
feature_extractor = FeatureExtractor.adjective_bag_of_words
corpus = Corpus.movie_review
classifier = Classifier.naive_bays