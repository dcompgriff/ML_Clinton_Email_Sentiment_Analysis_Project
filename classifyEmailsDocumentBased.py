import sentimentAnalysisDocumentBased
import generalStatistics
import dataLoadModule

from nltk.sentiment import SentimentAnalyzer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import nltk
import pandas as pd

def get_best_classifer_from_movies():
    """
    The best classifier for movies turned out to be Bigram, Linear SVM
    :return: the trained classifier using the entire corpus
    """
    neg_docs, pos_docs = sentimentAnalysisDocumentBased.get_movie_corpus()
    train_docs = neg_docs + pos_docs

    # Set up the Sentiment Analyzer
    analyzer = SentimentAnalyzer()
    analyzer.add_feat_extractor(sentimentAnalysisDocumentBased.extract_sig_bigram_feats)
    train_feat = list(analyzer.apply_features(train_docs, labeled=True))
    classifier = SklearnClassifier(LinearSVC()).train(train_feat)

    return analyzer, classifier



def get_best_classifier_from_debates():
    """
    The best classifier for debates turned out to be Freq Dist, Logitic
    :return: the trained classifier using the entire corpus
    """
    neg_docs, pos_docs = sentimentAnalysisDocumentBased.get_political_debates()
    train_docs = neg_docs + pos_docs

    # Set up the Sentiment Analyzer
    analyzer = SentimentAnalyzer()
    analyzer.add_feat_extractor(sentimentAnalysisDocumentBased.extract_freq_dist)
    train_feat = list(analyzer.apply_features(train_docs, labeled=True))
    classifier = SklearnClassifier(LogisticRegression()).train(train_feat)

    return analyzer, classifier


def classify_email(analyzer, classifer, data):
    pos_negs = []

    for ind, row in data.iterrows():
        text = row["MetadataSubject"].lower() + row["RawText"].lower()
        words = nltk.word_tokenize(text)
        feats = analyzer.extract_features(words)
        pos_negs.append(classifer.classify(feats))

    pos_negs_df = pd.Series(pos_negs)
    return pd.concat([data, pos_negs_df], axis=1)

def count_terms_from_classified_data(classified_data):
    pos_count = len(classified_data[classified_data[0] == "pos"])
    neg_count = len(classified_data[classified_data[0] == "neg"])

    print("pos_count", pos_count, "neg_count", neg_count)

    figures = generalStatistics.get_political_figures()
    countries = generalStatistics.get_countries()

    figures_pos = {}
    figures_neg = {}

    countries_pos = {}
    countries_neg = {}

    for ind, row in classified_data.iterrows():
        subject_words = row["MetadataSubject"].lower()
        message_words = row["RawText"].lower()

        for figure in figures:
            if figure + " " in (subject_words + message_words):
                if row[0] == "pos":
                    if figure in figures_pos:
                        figures_pos[figure] += 1
                    else:
                        figures_pos[figure] = 1
                else:
                    if figure in figures_neg:
                        figures_neg[figure] += 1
                    else:
                        figures_neg[figure] = 1
        for country in countries:
            if country in (subject_words + message_words):
                if row[0] == "pos":
                    if country in countries_pos:
                        countries_pos[country] += 1
                    else:
                        countries_pos[country] = 1
                else:
                    if country in countries_neg:
                        countries_neg[country] += 1
                    else:
                        countries_neg[country] = 1

    return (pd.DataFrame.from_dict(figures_pos, orient="index"), pd.DataFrame.from_dict(figures_neg, orient="index"),
        pd.DataFrame.from_dict(countries_pos, orient="index"), pd.DataFrame.from_dict(countries_neg, orient="index"))


def run_through_movies(data):
    analyzer, classifier = get_best_classifer_from_movies()
    classified_data = classify_email(analyzer, classifier, data)
    fp, fn, cp, cn = count_terms_from_classified_data(classified_data)
    politicans = fp.merge(fn, left_index=True, right_index=True).sort("0_x", ascending=False)[0:10]
    politicans.columns = ["pos", "neg"]
    countries = cp.merge(cn, left_index=True, right_index=True).sort("0_x", ascending=False)[0:10]
    countries.columns = ["pos", "neg"]

    return politicans, countries

def run_through_debates(data):
    analyzer, classifier = get_best_classifier_from_debates()
    classified_data = classify_email(analyzer, classifier, data)
    fp, fn, cp, cn = count_terms_from_classified_data(classified_data)
    politicans = fp.merge(fn, left_index=True, right_index=True).sort("0_x", ascending=False)[0:10]
    politicans.columns = ["pos", "neg"]
    countries = cp.merge(cn, left_index=True, right_index=True).sort("0_x", ascending=False)[0:10]
    countries.columns = ["pos", "neg"]

    return politicans, countries

politicians, countries = run_through_debates(dataLoadModule.getFullEmailData())
