import logging
import sklearn

logger = logging.getLogger(__name__)

class DenseTransformer(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()
