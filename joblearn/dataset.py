import logging
import os
import json
import requests
import re
import numpy as np
# import time
# import collections
# import pandas as pd
import scipy
import sklearn.datasets


logger = logging.getLogger(__name__)


class JobAdParagraphDataset():

    _data_dir = "./data"
    _api_url = "http://thesis.cwestrup.de/api"

    def __init__(self, file=None, timestamped=False):

        # timestamp for persisting data
        self.timestamp = ""
        # all labels used
        self.target_names = []
        # all processed documents in full text
        self.data = []
        # target matrix (csr_matrix)
        self.target = None

        # persist data files with timestamp if desired
        if timestamped:
            self.timestamp = "-" + str(datetime.datetime.now())

        # check if json file with chunked labelled job ads was provided,
        # otherwise download new data from API
        if not file or not os.path.exists(file):
            logger.info("No json data file provided/found, "+
                        "retrieving data from API.")
            file = self._download_json_data()
        with open(file) as data_file:
            data = json.load(data_file)

            (self.data,
             self.target_names,
             self.target) = self._convert_data(data)


    def _download_json_data(self):
        """ Retrieve json data from API """

        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        logger.info("Retrieved 0 tagged text chunks. Working...")

        url = (self._api_url + "/tags/populated");
        file = (self._data_dir + "/json_data_from_api"
                + self.timestamp + ".json")
        last_batch_received = False
        size = 100
        page = 1

        with open(file, 'w', encoding='utf8') as json_file:
            json_file.write("[")

        while not last_batch_received:
            params = {'size': size, 'page': page}
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error("Request not successful, " +
                             "cannot retrieve data from API")
                break
            # check if it's the last batch
            json = response.json()
            if (len(json) < size): last_batch_received = True
            data = response.text.lstrip("[").rstrip("]")
            with open(file, 'a', encoding='utf8') as json_file:
                json_file.write(data)
                if not last_batch_received:
                    json_file.write(', ')
                    logger.info("Retrieved " + str(page*size) +
                                " tagged text chunks. Working...")
                    page += 1
                else:
                    json_file.write("]")
                    logger.info("Retrieved " + str((page-1)*size + len(json)) +
                                " tagged text chunks. Done!")
        logger.info("Json data written to file " + file)
        return file


    def _convert_data(self, data):
        """ Store internal representation of documents and labels
            and a mapping between these.
        """

        documents = []
        labels = []

        # collect data for sparse coo matrix
        mapping_data = []
        mapping_row = []
        mapping_col = []

        for json_doc in data:
            for chunk in json_doc['chunks']:
                # seperate tags that used the hash sign and semicolon
                tags = re.split(r", |,| #|; ", json_doc['content'].strip("#"))
                document = chunk['_chunk']['content']
                # add new document if not stored yet
                try:
                    doc_index = documents.index(document)
                except ValueError:
                    documents.append(document)
                    doc_index = len(documents)-1
                for tag in tags:
                    # add new tag/label if not stored yet
                    try:
                        label_index = labels.index(tag)
                    except ValueError:
                        labels.append(tag)
                        label_index = len(labels)-1
                    # add mapping between document and labels
                    # (if it's not already set)
                    if not (doc_index in mapping_row
                            and label_index in mapping_col):
                        mapping_data.append(1)
                        mapping_row.append(doc_index)
                        mapping_col.append(label_index)

        # Create the COO-matrix
        coo = scipy.sparse.coo_matrix((mapping_data,(mapping_row,mapping_col)),
                                dtype=np.int)
        # Convert to csr matrix
        docs_x_labels = scipy.sparse.csr_matrix(coo)

        return documents, labels, docs_x_labels


    def load(self):
        return sklearn.datasets.base.Bunch(data=self.data,
                     target=self.target.toarray(),
                     target_names=self.target_names)
