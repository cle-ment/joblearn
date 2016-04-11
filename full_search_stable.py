import pickle
import datetime
import logging

import joblearn.dataset
import joblearn.gridsearch
import joblearn.feat_extr
import joblearn.feat_trans
import joblearn.scoring
import joblearn.target_trans
import joblearn.estimation

import sklearn.cross_validation
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.tree
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.ensemble


### Basic setup

# test set size
TEST_SIZE = 0.3
# timestamp at runtime
TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")


### Logger setup

# create logger with 'joblearn'
logger = logging.getLogger('joblearn')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('logs/full_search_stable-' + TIMESTAMP + '.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(name)s:%(levelname)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


### Scores setup

scores = {}

scores['log_loss'] = joblearn.scoring.ScoreSetting(
    "Cross-entropy",'log_loss')

scores['f1_micro'] = joblearn.scoring.ScoreSetting(
    "F1 micro", 'f1_micro')

scores['mostlikely'] = joblearn.scoring.ScoreSetting(
    "Most likely class", joblearn.scoring.mostlikely_score)


### Dataset Initialization

jobad_dataset = joblearn.dataset.JobAdParagraphDataset(
    "./data/json_data_from_api.json").load()
data_X = jobad_dataset.data
data_Y = jobad_dataset.target
data_Y_labels = jobad_dataset.target_names


### Label Groupings Setup

label_groupings = {}

## Manually grouped
with open('data/labels_grouped.pickle', 'rb') as handle:
    labels_grouped = pickle.load(handle)
(Y, Y_labels) = joblearn.target_trans.group_labels(data_Y, data_Y_labels,
                                                   labels_grouped)
title = "Manually grouped (" + str(len(Y_labels)) + " labels)"
label_groupings["grouped"] = joblearn.target_trans.LabelGrouping(title, Y,
                                                                 Y_labels)

## Manually structured
with open('data/labels_structured.pickle', 'rb') as handle:
    labels_structured = pickle.load(handle)
(Y, Y_labels) = joblearn.target_trans.group_labels(data_Y, data_Y_labels,
                                                   labels_structured)
title = "Manually structured (" + str(len(Y_labels)) + " labels)"
label_groupings["structured"] = joblearn.target_trans.LabelGrouping(title, Y,
                                                                    Y_labels)

## Clustered
# TODO


### Train/Test Splits Setup

data_splits = {}

for key in label_groupings:
    (X_train, X_test, Y_train,
     Y_test) = sklearn.cross_validation.train_test_split(
         data_X, label_groupings[key].Y, test_size=TEST_SIZE, random_state=0)
    data_splits[key] = joblearn.target_trans.DataSplit(X_train, X_test,
                                                       Y_train, Y_test)


### Feature Extraction Setup

feature_extractors = {}

## N-gram Words, TF.IDF weighing

title = "N-gram Words, TF.IDF weighing"
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
    analyzer='word', max_features=200)
parameter_space = {
        'vect__ngram_range': [(1,1), (1,2), (1,3)],
        'vect__stop_words': ['english', None],
        'vect__sublinear_tf': [True, False]
    }
feature_extractors['ngram_words'] = joblearn.feat_extr.FeatureExtractorSetting(
    title, vectorizer, parameter_space)

## N-gram Chars, TF.IDF weighing

title = "N-gram Chars, TF.IDF weighing"
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
    analyzer='char', max_features=200)
parameter_space = {
        'vect__ngram_range': [(1,10), (1,20), (10,20)],
        'vect__stop_words': ['english', None],
        'vect__sublinear_tf': [True, False]
    }
feature_extractors['ngram_chars'] = joblearn.feat_extr.FeatureExtractorSetting(
    title, vectorizer, parameter_space)

## Word2Vec
# TODO


### Estimator Setup

estimators = {}

## Logistic Regression

title = "Logistic Regression, one-vs-rest"
classifier = sklearn.linear_model.LogisticRegression()
estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
parameter_space = {}
estimators['log-regr_1vR'] = joblearn.estimation.EstimatorSetting(
    title, estimator, parameter_space)

# ## Decision Tree
#
# title = "Decision Tree, one-vs-rest"
# classifier = sklearn.tree.DecisionTreeClassifier()
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# parameter_space = {}
# estimators['decisiontree_1vR'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)
#
# ## Naive Bayes
#
# title = "Naive Bayes, one-vs-rest"
# classifier = sklearn.naive_bayes.MultinomialNB()
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# parameter_space = {}
# estimators['naive-bayes_1vR'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)
#
# ## SVM
#
# title = "SVM, linear kernel, one-vs-rest"
# classifier = sklearn.svm.SVC(probability=True)
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# parameter_space = {'clf__estimator__kernel': ['linear'],
#                    'clf__estimator__C': [1, 10, 100, 1000]}
# estimators['svm_lin_1vR'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)
#
# title = "SVM, rbf kernel, one-vs-rest"
# classifier = sklearn.svm.SVC(probability=True)
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# parameter_space = {'clf__estimator__kernel': ['rbf'],
#                    'clf__estimator__gamma': [1e-3, 1e-2, 1e-1, 1],
#                    'clf__estimator__C': [1, 10, 100, 1000]}
# estimators['svm_rbf_1vR'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)
#
# ## KNN
#
# title = "KNN, one-vs-rest"
# classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# parameter_space = {'clf__estimator__weights': ['uniform', 'distance']}
# estimators['knn_1vR'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)
#
# ## Random Forest
#
# title = "Random Forest, one-vs-rest"
# classifier = sklearn.ensemble.RandomForestClassifier()
# estimator = sklearn.multiclass.OneVsRestClassifier(classifier)
# parameter_space = {}
# estimators['randomforest_1vR'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)


### Exhaustive grid search

grid_cv = joblearn.gridsearch.GridSearch()
grid_cv.run(label_groupings, data_splits, feature_extractors,
                     estimators, scores, n_jobs=-1, 
                     to_csv="results/" + EXP_NAME + "-" + TIMESTAMP + ".csv");
