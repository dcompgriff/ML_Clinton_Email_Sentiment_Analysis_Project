import constants
import sentimentAnalysisDocumentBased

import itertools

if __name__ == "__main__":
    for corpus, f_extractor, classifier in itertools.product(
            constants.Corpus, constants.FeatureExtractor, constants.Classifier):
        print(corpus, f_extractor, classifier)

        constants.corpus = corpus
        constants.feature_extractor = f_extractor
        constants.classifier = classifier

        sentimentAnalysisDocumentBased.main()