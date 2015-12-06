import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import nltk.sentiment.util
from nltk.sentiment import SentimentAnalyzer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import nltk.data, nltk.tag

import sklearn.cross_validation

import itertools

import constants

"""
Corpus
"""
def get_movie_corpus():
    from nltk.corpus import movie_reviews
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
    negdocs = [(list(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posdocs = [(list(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    return negdocs, posdocs


"""
Feature Extractors
"""
def extract_bag_of_words_feats(word_list):
    """
    A simple bag-of-words feature extractor.  Takes a document and finds the bag-of-words, which is simply a dict of the
    words in the document mapped to True.
    :param word_list: a list of words
    :return: a dictionary
    """
    return dict([(word, True) for word in word_list])

def extract_big_gram_feats(word_list):
    bigram_finder = BigramCollocationFinder.from_words(word_list)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 150)

    # Return both the bag_of_words + the bigrams
    return dict([(ngram, True) for ngram in itertools.chain(word_list, bigrams)])

def adjective_bag_of_words(word_list):
    global tagger
    word_pos = tagger.tag(word_list)
    return dict([(word, True) for word, pos in word_pos if pos == "JJ"])



if __name__ == "__main__":
    if constants.corpus == constants.Corpus.movie_review:
        neg_docs, pos_docs = get_movie_corpus()

    if constants.mark_negation:
        neg_docs = [nltk.sentiment.util.mark_negation(doc) for doc in neg_docs]
        pos_docs = [nltk.sentiment.util.mark_negation(doc) for doc in pos_docs]

    # Split betweeen the training set and the testing set
    num_train_neg = int(3/4*len(neg_docs)); num_test_neg = len(neg_docs) - num_train_neg
    num_train_pos = int(3/4*len(pos_docs)); num_test_pos = len(pos_docs) - num_train_pos

    train_neg, test_neg = sklearn.cross_validation.train_test_split(neg_docs, train_size=num_train_neg,
                                                                    test_size=num_test_neg)
    train_pos, test_pos = sklearn.cross_validation.train_test_split(pos_docs, train_size=num_train_pos,
                                                                    test_size=num_test_pos)

    # Make the final train set and test set
    train_docs = train_pos + train_neg
    test_docs = test_pos + test_neg

    # Set up the Sentiment Analyzer
    analyzer = SentimentAnalyzer()
    all_words = analyzer.all_words(train_docs, labeled=True)

    if constants.feature_extractor == constants.FeatureExtractor.bag_of_words:
        analyzer.add_feat_extractor(extract_bag_of_words_feats)
    elif constants.feature_extractor == constants.FeatureExtractor.unigram:
        unigram_features = analyzer.unigram_word_feats(all_words, min_freq=1000)
        print("Length of unigram features: %d" % len(unigram_features))
        analyzer.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigram_features)
    elif constants.feature_extractor == constants.FeatureExtractor.big_gram:
        analyzer.add_feat_extractor(extract_big_gram_feats)
    elif constants.feature_extractor == constants.FeatureExtractor.adjective_bag_of_words:
        tagger = nltk.tag.HunposTagger()
        analyzer.add_feat_extractor(adjective_bag_of_words)

    train_feat = list(analyzer.apply_features(train_docs, labeled=True))
    test_feat = list(analyzer.apply_features(test_docs, labeled=True))

    print('train on %d instances, test on %d instances' % (len(train_feat), len(test_feat)))

    if constants.classifier == constants.Classifier.naive_bays:
        classifier = NaiveBayesClassifier.train(train_feat)
        print('accuracy:', nltk.classify.util.accuracy(classifier, test_feat))
        classifier.show_most_informative_features()