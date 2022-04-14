import logging
from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from infercode.client.base_client import BaseClient
from infercode.network.infercode_network import InferCodeModel

tf.compat.v1.disable_v2_behavior()


class InferCodeClient(BaseClient):
    logger = logging.getLogger('InferCodeTrainer')

    def __init__(self, language, config=None):
        super().__init__(language, config)

    def init_from_config(self):
        # ------------Set up the neural network------------
        self.infercode_model = InferCodeModel(num_types=self.node_type_vocab.get_vocabulary_size(),
                                              num_tokens=self.node_token_vocab.get_vocabulary_size(),
                                              num_subtrees=self.subtree_vocab.get_vocabulary_size(),
                                              num_languages=self.language_util.get_num_languages(),
                                              num_conv=self.num_conv,
                                              node_type_dim=self.node_type_dim,
                                              node_token_dim=self.node_token_dim,
                                              conv_output_dim=self.conv_output_dim,
                                              include_token=self.include_token,
                                              batch_size=self.batch_size,
                                              learning_rate=self.learning_rate)

        self.saver = tf.compat.v1.train.Saver(save_relative_paths=True, max_to_keep=5)
        self.init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init)

        self.checkfile = self.model_checkpoint / 'cnn_tree.ckpt'
        ckpt = tf.train.get_checkpoint_state(self.model_checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            self.logger.info("Load model successfully : " + str(self.checkfile))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            error_message = "Could not find the model : " + str(self.checkfile)
            self.logger.error(error_message)
            raise ValueError(error_message)

    def snippets_to_tensors(self, batch_code_snippets: Iterable[Union[str, bytes]]):
        batch_tree_indexes = []
        assert len(batch_code_snippets) <= 5
        for code_snippet in batch_code_snippets:
            # tree-sitter parser requires bytes as the input, not string
            if isinstance(code_snippet, str):
                code_snippet = str.encode(code_snippet)
            ast = self.ast_parser.parse(code_snippet)
            tree_representation, _ = self.ast_util.simplify_ast(ast, code_snippet)
            tree_indexes = self.tensor_util.transform_tree_to_index(tree_representation)
            batch_tree_indexes.append(tree_indexes)

        tensors = self.tensor_util.trees_to_batch_tensors(batch_tree_indexes)
        return tensors

    def encode(self, batch_code_snippets: Iterable[Union[str, bytes]]) -> np.ndarray:
        tensors = self.snippets_to_tensors(batch_code_snippets)
        embeddings = self.sess.run(
            [self.infercode_model.code_vector],
            feed_dict={
                self.infercode_model.placeholders["node_type"]: tensors["batch_node_type_id"],
                self.infercode_model.placeholders["node_tokens"]: tensors["batch_node_tokens_id"],
                self.infercode_model.placeholders["children_index"]: tensors["batch_children_index"],
                self.infercode_model.placeholders["children_node_type"]: tensors["batch_children_node_type_id"],
                self.infercode_model.placeholders["children_node_tokens"]: tensors["batch_children_node_tokens_id"],
                self.infercode_model.placeholders["language_index"]: self.language_index,
                self.infercode_model.placeholders["dropout_rate"]: 0.0
            }
        )
        return embeddings[0]
