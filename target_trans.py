import logging
import collections
import numpy as np

logger = logging.getLogger(__name__)

LabelGrouping = collections.namedtuple('LabelGrouping',
                                       ['title', 'Y', 'Y_labels', ])

DataSplit = collections.namedtuple('DataSplit',
                                   [ 'X_train', 'X_test', 'Y_train', 'Y_test'])

def group_labels(Y, target_names, label_group_dict):
    """ Create a grouping between the labels i.e. map several labels
        to the same value.
        Expects a grouping dict with all labels, of the form e.g.
            {0: ['job', 'time frame'],
             1: ['further information',
                 'contact information']
            ...
            }
        The keys of the dictionary can also be strings which results in a
        renaming of the group instead of a concatenation of the names.
    """
    new_column_arrays = []
    new_labels = []
    for key, labels in label_group_dict.items():
        # if a new name was given for the label group then use that
        if type(key) == str:
            new_labels.append(key)
        # otherwise use the stringified the list of labels for that group
        else:
            new_labels.append(str(labels))
        label_ids = []
        # collect id's for labels to be joined
        for label in labels:
            try:
                label_ids.append(target_names.index(label))
            except ValueError:
                logger.debug("Label '" + label + "' not found in labels, "+
                              "skipping.")
        # create new label by taking the max from all labels to be joined
        try:
            new_column_arrays.append(Y[:,label_ids].max(axis=1, keepdims=True))
        except (ValueError, IndexError):
            # No labels found in this label group, skip this group
            pass

    return (np.hstack(new_column_arrays), new_labels)

def make_grouping(cluster_predictions, target_names):
    clustered_tags = {}
    for i, label in enumerate(target_names):
        cluster_idx = int(cluster_predictions)
        try:
            clustered_tags[cluster_idx].append(label)
        except KeyError:
            clustered_tags[cluster_idx] = [label]
