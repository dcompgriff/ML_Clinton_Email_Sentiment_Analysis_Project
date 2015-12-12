import pandas as pd
import nltk

import dataLoadModule
import constants

def get_countries():
    countries = set()

    with open(constants.country_list, 'r') as f:
        for country in f.readlines():
            countries.add(country.strip())

    return countries


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


def find_mentioned_countries(data):
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

    return countries_mentioned


data = dataLoadModule.getFullEmailData()
countries = find_mentioned_countries(data)
print(countries)
