import logging
import math
import random
from collections import defaultdict

import tensorflow as tf

from infercode.data_utils.tensor_util import TensorUtil


class DataSequence(tf.keras.utils.Sequence):
    """https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence"""
    logger = logging.getLogger('DataSequence')

    def __init__(self, batch_size: int, buckets: defaultdict, tree_max_size: int = 500):
        self.batch_size = batch_size
        self.tensor_util = TensorUtil()
        self.buckets = buckets
        self.bucket_trees = self._filter_bucket_trees(tree_max_size)
        self.trees = self._flatten_bucket_trees()
        self._shuffle_bucket_trees()  # shuffle data before first epoch

    def __len__(self):
        """steps per epoch"""
        return math.ceil(len(self.trees) / self.batch_size)

    def __getitem__(self, index):
        batch_trees = self.trees[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x, batch_y = self.tensor_util.trees_to_batch_tensors(batch_trees)
        return batch_x, batch_y

    def on_epoch_end(self):
        """shuffle and flatten the data after each epoch"""
        self._shuffle_bucket_trees()

    def _filter_bucket_trees(self, max_size: int) -> list[list]:
        filter_bucket_trees = []
        bucket_trees = list(self.buckets.values())
        random.shuffle(bucket_trees)
        for trees in bucket_trees:
            filter_bucket_trees.append([tree for tree in trees if tree["size"] < max_size])
        return filter_bucket_trees

    def _flatten_bucket_trees(self) -> list:
        trees = []
        for bucket_trees in self.bucket_trees:
            trees.extend(bucket_trees)
        return trees

    def _shuffle_bucket_trees(self) -> None:
        random.shuffle(self.bucket_trees)
        for trees in self.bucket_trees:
            random.shuffle(trees)
        self.trees = self._flatten_bucket_trees()
