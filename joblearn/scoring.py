import logging
import sklearn
import collections
import sklearn.metrics
import numpy as np

logger = logging.getLogger(__name__)

ScoreSetting = collections.namedtuple('ScoreSetting', ['title', 'score'])

def intersection_any_score_func(ground_truth, predictions):
    n_predictions = predictions.shape[0]
    # correct if same label in prediction and ground truth
    correct_predictions = (predictions + ground_truth) > 1
    # success = there was at least one correctly predicted label
    successes = correct_predictions.max(axis=1)
    return sum(successes) / n_predictions

intersection_any_score = sklearn.metrics.make_scorer(intersection_any_score_func)

def intersection_all_score_func(ground_truth, predictions):
    n_predictions = predictions.shape[0]
    # correct if same label in prediction and ground truth
    correct_predictions = (predictions + ground_truth) > 1
    # ... and if the number of predicted labels is correct
    correct_num_labels = (correct_predictions.sum(axis=1)
                          == ground_truth.sum(axis=1))
    # success = there was at least one correctly predicted label
    successes = correct_predictions.max(axis=1)
    return sum(np.all([correct_num_labels,successes], axis=0)) / n_predictions

intersection_all_score = sklearn.metrics.make_scorer(intersection_all_score_func)

def mostlikely_score(estimator, X, y):
    y_pred = estimator.predict_proba(X)
    y_pred_ml = np.argmax(y_pred, axis=1)
    true_predictions = 0
    for y_row, most_likely_label in zip((y == 1), y_pred_ml):
        # get the list of true labels
        true_labels = np.nonzero(y_row)[0]
        # if the most likely label is in the true labels it's a success
        if most_likely_label in true_labels:
            true_predictions += 1
    return true_predictions/len(y)
