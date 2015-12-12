import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer

import dataLoadModule
import constants

def get_countries():
    countries = set()

    with open(constants.country_list, 'r') as f:
        for country in f.readlines():
            countries.add(country.strip())

    return countries

def get_political_figures():
    political_figures = set()

    with open(constants.political_figures, 'r', encoding="utf-8") as f:
        for political_figure in f.readlines():
            political_figures.add(political_figure.strip())

    return political_figures


def filter_significant_email(data):
    """
    We'll define any significant email any emails containg 500 characters or more.

    :param data: The Pandas dataframe containing all of Hillary Clinton's data
    :return: a filtered dataframe
    """
    frame_to_choose = []
    for ind, row in data.iterrows():
        if len(row["RawText"]) > 500:
            frame_to_choose.append(True)
        else:
            frame_to_choose.append(False)

    return data[frame_to_choose]


def count_mentioned_countries(data):
    """
    Return the frequency of countries mentioned in Clinton's emails
    (is a map from country to integer how many emails the country is mentioned)

    :param data: email data as a Pandas data frame
    :return: Countries mentioned in the email
    """
    countries_mentioned = {}
    countries = get_countries()

    for ind, row in data.iterrows():
        subject_words = row["MetadataSubject"].lower()
        message_words = row["RawText"].lower()

        for country in countries:
            if country in (subject_words + message_words):
                if country in countries_mentioned:
                    countries_mentioned[country] += 1
                else:
                    countries_mentioned[country] = 1

    return pd.DataFrame.from_dict(countries_mentioned, orient="index")


def count_mentioned_pol_figures(data):
    """
    Return the frequency of political figures mentioned in Clinton's emails
    (is a map from country to integer how many emails the country is mentioned)

    :param data: email data as a Pandas data frame
    :return: pol figures mentioned in the email
    """
    figures_mentioned = {}
    figures = get_political_figures()

    for ind, row in data.iterrows():
        subject_words = row["MetadataSubject"].lower()
        message_words = row["RawText"].lower()

        for figure in figures:
            if figure + " " in (subject_words + message_words):
                if figure in figures_mentioned:
                    figures_mentioned[figure] += 1
                else:
                    figures_mentioned[figure] = 1

    return pd.DataFrame.from_dict(figures_mentioned, orient="index")

def basic_statistics_of_email(data):
    """
    Print basic statistics of the email data

    :param data: Pandas dataframe for email data
    :return: None
    """
    word_counts = []
    character_count = 0

    for ind, row in data.iterrows():
        tokenizer = RegexpTokenizer(r'\w+')
        real_words = tokenizer.tokenize(row["RawText"].lower())

        character_count += sum(map(len, real_words))
        word_counts.append(len(real_words))

    return character_count, pd.Series(word_counts)




data = dataLoadModule.getFullEmailData()
c_count, w_count = basic_statistics_of_email(data)
