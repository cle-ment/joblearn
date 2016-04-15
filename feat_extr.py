import sklearn
import logging
import collections

logger = logging.getLogger(__name__)

FeatureExtractorSetting = collections.namedtuple('FeatureExtractorSetting', 
                              ['title', 'vectorizer', 'parameter_space'])

class Word2VecVectorizer(sklearn.base.TransformerMixin):

    def __init__(self, pretrained_model_file):

        # load pretrained word2vec model from file
        self.model = gensim.models.Word2Vec.load_word2vec_format(
            pretrained_model_file, binary=True)

    def vectorize_docs_mean(self, X, y):
        n_features = Word2VecModelMean.model.vector_size
        docs_x_features_rows = []
        documents_unable_to_vectorize = []
        for i, document in enumerate(documents):
            document_words = document.split()
            vectors = []
            for word in document_words:
                try:
                    vectors.append(self.model[word])
                except KeyError:
                    pass # word not found in word2vec vocab
            if vectors:
                docs_x_features_rows.append(np.average(vectors, axis=0)
                                            .reshape((1,n_features)))
            else:
                # none of the words found in word2vec vocab, remove the document
                documents_unable_to_vectorize.append(i)
        return docs_x_features_rows, documents_unable_to_vectorize

    # fit does nothing (because we start with a pretrained model)
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y, **transform_params):

        (docs_x_features_rows,
         documents_unable_to_vectorize) = self.vectorize_docs_mean(X, y)

        # remove documents that could not be vectorized
        remove_indexes = sorted(list(documents_unable_to_vectorize), reverse=True)
        logger.warning("Some documents only contained words that could not be "+
                       "found in the word2vec vocabulary. These will be removed. ")
        logger.debug("The following documents will be removed: ")
        for index in remove_indexes:
            logger.debug(str(self.documents[index]))
            del self.documents[index]

        # also delete these documents from docs_x_labels
        num_docs = self.docs_x_labels.shape[0]
        mask = np.ones(num_docs, np.bool)
        mask[remove_indexes] = 0
        self.docs_x_labels = sparse.csr_matrix(self.docs_x_labels)[mask,:]

        return sparse.csr_matrix(np.vstack(docs_x_features_rows))
