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

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.multiclass
import sklearn.metrics

import skflow
import tensorflow as tf
import numpy as np



### Basic setup

# test set size
TEST_SIZE = 0.3
# timestamp at runtime
TIMESTAMP = str(datetime.datetime.now())
EXP_NAME = __file__.rstrip(".py")

# logger
logging.basicConfig(format='%(asctime)s %(name)s:%(levelname)s: %(message)s ')
logger = logging.getLogger("joblearn")
logger.setLevel(logging.INFO)
fh = logging.FileHandler('gridsearch.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)


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


## Manually structured
with open('data/labels_structured.pickle', 'rb') as handle:
    labels_structured = pickle.load(handle)
(Y, Y_labels) = joblearn.target_trans.group_labels(data_Y, data_Y_labels,
                                                   labels_structured)
title = "Manually structured (" + str(len(Y_labels)) + " labels)"
label_groupings["structured"] = joblearn.target_trans.LabelGrouping(title, Y,
                                                                    Y_labels)


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
    analyzer='word',
    max_features=20)
parameter_space = {
        # 'vect__ngram_range': [(1,1), (1,2), (1,3)],
        # 'vect__stop_words': ['english', None],
        # 'vect__sublinear_tf': [True, False]
    }
feature_extractors['ngram_words'] = joblearn.feat_extr.FeatureExtractorSetting(
    title, vectorizer, parameter_space)


### Estimator Setup

estimators = {}

title = "TensorFlow NN"
num_classes = len(label_groupings['structured'].Y_labels)

MAX_DOCUMENT_LENGTH = 100

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models

EMBEDDING_SIZE = 20
N_FILTERS = 10
WINDOW_SIZE = 20
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

def cnn_model(X, y):
    """2 layer Convolutional network to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # Apply Convolution filtering on input sequence.
        conv1 = skflow.ops.conv2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # Add a RELU for non linearity.
        conv1 = tf.nn.relu(conv1)
        # Max pooling across output of Convlution+Relu.
        pool1 = tf.nn.max_pool(conv1, ksize=[1, POOLING_WINDOW, 1, 1],
            strides=[1, POOLING_STRIDE, 1, 1], padding='SAME')
        # Transpose matrix so that n_filters from convolution becomes width.
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # Second level of convolution filtering.
        conv2 = skflow.ops.conv2d(pool1, N_FILTERS, FILTER_SHAPE2,
            padding='VALID')
        # Max across each filter to get useful features for classification.
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])
    # Apply regular WX + B and classification.
    return skflow.models.logistic_regression(pool2, y)


classifier = skflow.TensorFlowEstimator(model_fn=cnn_model,
                                        n_classes=num_classes,
    steps=100, optimizer='Adam', learning_rate=0.01, continue_training=True)

estimator = sklearn.multiclass.OneVsRestClassifier(classifier)


# Continuesly train for 1000 steps & predict on test set.
while True:
    estimator.fit(X_train, Y_train)
    score = sklearn.metrics.accuracy_score(Y_test, estimator.predict(X_test))
    print('Accuracy: {0:f}'.format(score))

# classifier = skflow.TensorFlowDNNClassifier(
#     hidden_units=[20, 40],
#     n_classes=num_classes,
#     learning_rate=0.1)

# scaler = sklearn.preprocessing.StandardScaler()
# pipeline = sklearn.pipeline.Pipeline(
#     [('dense', joblearn.feat_trans.DenseTransformer()),
#      ('scaler', scaler),
#      ('clf', classifier)])


# parameter_space = {}
# estimator = sklearn.multiclass.OneVsRestClassifier(pipeline)
# estimators['tf_nn'] = joblearn.estimation.EstimatorSetting(
#     title, estimator, parameter_space)


### Exhaustive grid search
#
# grid_cv = joblearn.gridsearch.GridSearch()
# grid_cv.run(label_groupings, data_splits, feature_extractors,
#                      estimators, scores, n_jobs=1, cv=4,
#                      to_csv="./results/" + EXP_NAME + "-" + TIMESTAMP + ".csv");
