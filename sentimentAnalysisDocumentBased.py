import nltk.classify.util
from nltk.classify import NaiveBayesClassifier, MaxentClassifier, DecisionTreeClassifier
import nltk.sentiment.util
from nltk.sentiment import SentimentAnalyzer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import nltk.data, nltk.tag
from nltk.classify.scikitlearn import SklearnClassifier
import nltk

import sklearn.cross_validation
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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


def get_political_debates():
    from glob import glob
    pos_files = [file for file in glob("data/training_data/political_debates/*") if file[-5] == "Y"]
    neg_files = [file for file in glob("data/training_data/political_debates/*") if file[-5] == "N"]

    neg_docs = []
    pos_docs = []

    for pos_file in pos_files:
        with open(pos_file, 'r', encoding='utf-8') as f:
            pos_doc = f.read()
            pos_docs.append((nltk.word_tokenize(pos_doc.lower()), 'pos'))
    for neg_file in neg_files:
        with open(neg_file, 'r', encoding='utf-8') as f:
            neg_doc = f.read()
            neg_docs.append((nltk.word_tokenize(neg_doc.lower()), 'neg'))

    return neg_docs, pos_docs

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

def extract_freq_dist(word_list):
    """
    A simple frequency feature extractor.  Takes a document and finds the frequency distribution, which is simply a dict
    words in the document mapped to the number of instances of the words.
    :param word_list: a list of words
    :return: a dictionary
    """
    ret_dict = {}
    for word in word_list:
        if word in ret_dict:
            ret_dict[word] += 1
        else:
            ret_dict[word] = 1

    return ret_dict

def extract_sig_bigram_feats(word_list):
    bigram_finder = BigramCollocationFinder.from_words(word_list)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 150)

    # Return both the bag_of_words + the bigrams
    return dict([(ngram, True) for ngram in itertools.chain(word_list, bigrams)])

def adjective_bag_of_words(word_list):
    global tagger
    word_pos = tagger.tag(word_list)
    return dict([(word, True) for word, pos in word_pos if pos == b"JJ"])


def adj_noun_adv_vb_bag_of_words(word_list):
    global tagger
    word_pos = tagger.tag(word_list)
    return dict([(word, True) for word, pos in word_pos if pos == b"JJ" or pos == b"NN" or pos == b"RB" or pos == b"VB"])


if __name__ == "__main__":
    if constants.corpus == constants.Corpus.movie_review:
        neg_docs, pos_docs = get_movie_corpus()
    elif constants.corpus == constants.Corpus.pol_debates:
        neg_docs, pos_docs = get_political_debates()

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
    elif  constants.feature_extractor == constants.FeatureExtractor.freq_dist:
        analyzer.add_feat_extractor(extract_freq_dist)
    elif constants.feature_extractor == constants.FeatureExtractor.unigram:
        unigram_features = analyzer.unigram_word_feats(all_words, min_freq=1000)
        print("Length of unigram features: %d" % len(unigram_features))
        analyzer.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigram_features)
    elif constants.feature_extractor == constants.FeatureExtractor.bigram_bag_of_words:
        analyzer.add_feat_extractor(extract_sig_bigram_feats)
    elif constants.feature_extractor == constants.FeatureExtractor.adjective_bag_of_words:
        tagger = nltk.tag.HunposTagger(constants.hunpos_english_model)
        analyzer.add_feat_extractor(adjective_bag_of_words)
    elif constants.feature_extractor == constants.FeatureExtractor.pos_bag_of_words:
        tagger = nltk.tag.HunposTagger(constants.hunpos_english_model)
        analyzer.add_feat_extractor(adjective_bag_of_words)

    train_feat = list(analyzer.apply_features(train_docs, labeled=True))
    test_feat = list(analyzer.apply_features(test_docs, labeled=True))

    print('train on %d instances, test on %d instances' % (len(train_feat), len(test_feat)))

    if constants.classifier == constants.Classifier.naive_bays:
        classifier = NaiveBayesClassifier.train(train_feat)
        analyzer.evaluate(test_feat, classifier, accuracy=True, f_measure=True, precision=True, recall=True,
                          verbose=True)
        classifier.show_most_informative_features()
    elif constants.classifier == constants.Classifier.maxent:
        classifier = MaxentClassifier.train(train_feat)
        analyzer.evaluate(test_feat, classifier, accuracy=True, f_measure=True, precision=True, recall=True,
                          verbose=True)
        classifier.show_most_informative_features()
    elif constants.classifier == constants.Classifier.decision_tree:
        classifier =  SklearnClassifier(DecisionTreeClassifier()).train(train_feat)
        analyzer.evaluate(test_feat, classifier, accuracy=True, f_measure=True, precision=True, recall=True,
                          verbose=True)
    elif constants.classifier == constants.Classifier.linear_svm:
        classifier = SklearnClassifier(LinearSVC()).train(train_feat)
        analyzer.evaluate(test_feat, classifier, accuracy=True, f_measure=True, precision=True, recall=True,
                          verbose=True)
    elif constants.classifier == constants.Classifier.random_forest:
        classifier = SklearnClassifier(RandomForestClassifier()).train(train_feat)
        analyzer.evaluate(test_feat, classifier, accuracy=True, f_measure=True, precision=True, recall=True,
                          verbose=True)
    elif constants.classifier == constants.Classifier.logistic:
        classifier = SklearnClassifier(LogisticRegression()).train(train_feat)
        analyzer.evaluate(test_feat, classifier, accuracy=True, f_measure=True, precision=True, recall=True,
                          verbose=True)