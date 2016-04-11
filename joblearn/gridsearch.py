import logging

import pandas as pd
import qgrid

import sklearn.pipeline
import sklearn.grid_search

logger = logging.getLogger(__name__)

class GridSearch():

    def __init__(self):

        self.results = {}

        self.columns = ['label_grouping_key',
                        'label_grouping_title',
                        'label_grouping_settings',
                        'feature_extractor_key',
                        'feature_extractor_title',
                        'feature_extractor_settings',
                        'estimator_key',
                        'estimator_title',
                        'estimator_settings',
                        'score_key',
                        'score_title',
                        'score_settings',

                        'score_train_',
                        'score_test_',
                        'best_estimator_',
                        'best_params_',
                        'grid_scores_']

    # results with the same label/feature/estimator/score configuration names
    # will be overwritten in self.results.
    def run(self, label_groupings, data_splits, feature_extractors, estimators,
            scores, cv=5, n_jobs=1, to_csv=None):

        self.label_groupings = label_groupings
        self.data_splits = data_splits
        self.feature_extractors = feature_extractors
        self.estimators = estimators
        self.scores = scores

        # total amount of grid search settings
        total_settings = (len(label_groupings)
                          * len(feature_extractors)
                          * len(estimators)
                          * len(scores))

        # the current the grid search settings
        current_setting = 0

        for l_key, l_settings in label_groupings.items():
            for f_key, f_settings in feature_extractors.items():
                for e_key, e_settings in estimators.items():
                    for s_key, s_settings in scores.items():

                        current_setting += 1

                        logger.info("Running setting "
                            + str(current_setting) + "/"
                            + str(total_settings) + ": "
                            + str(l_settings.title) + " | "
                            + str(f_settings.title) + " | "
                            + str(e_settings.title) + " | "
                            + str(s_settings.title))

                        pipeline = sklearn.pipeline.Pipeline([
                            ('vect', f_settings.vectorizer),
                            ('clf', e_settings.estimator)])

                        parameters = {}
                        parameters.update(f_settings.parameter_space)
                        parameters.update(e_settings.parameter_space)

                        grid_search_cv = sklearn.grid_search.GridSearchCV
                        grid_search = grid_search_cv(pipeline,
                                                     parameters,
                                                     scoring=s_settings.score,
                                                     cv=cv,
                                                     n_jobs=n_jobs)
                        grid_search.fit(self.data_splits[l_key].X_train,
                                        self.data_splits[l_key].Y_train)

                        score_train_ = grid_search.best_score_
                        score_test_  = grid_search.score(
                                            self.data_splits[l_key].X_test,
                                            self.data_splits[l_key].Y_test)

                        hash_key = hash(l_key + f_key + e_key + s_key)
                        self.results[hash_key] = [
                                    l_key, l_settings.title, l_settings,
                                    f_key, f_settings.title, f_settings,
                                    e_key, e_settings.title, e_settings,
                                    s_key, s_settings.title, s_settings,
                                    score_train_,
                                    score_test_,
                                    grid_search.best_estimator_,
                                    grid_search.best_params_,
                                    grid_search.grid_scores_]

                        # update results
                        resultrows = []
                        for key, result in self.results.items():
                            resultrows.append(result)

                        self.results_table = pd.DataFrame(resultrows,
                                                          columns=self.columns)
                        if (to_csv):
                            self.results_to_csv(to_csv)

        return self.results_table

    def results_to_csv(self, filename):
        self.results_table.to_csv(filename)

def show_results(results_table, short_titles=True):
    if short_titles:
        columns = ['label_grouping_key',
                     'feature_extractor_key',
                     'estimator_key',
                     'score_key',
                     'score_train_',
                     'score_test_']
    else:
        columns = ['label_grouping_title',
                     'feature_extractor_title',
                     'estimator_title',
                     'score_title',
                     'score_train_',
                     'score_test_']
    qgrid.show_grid(results_table[columns], precision=3,
                    grid_options={'enableColumnReorder': True})
