import logging

import tensorflow as tf

from infercode.network.layers import AggregationLayer, TBConvLayer


class InferCodeModel(tf.keras.Model):
    logger = logging.getLogger('InferCodeModel')

    def __init__(self, num_types, num_tokens, num_subtrees, num_languages, num_conv: int = 2,
                 node_type_dim: int = 50, node_token_dim: int = 50, conv_output_dim: int = 50,
                 include_token: bool = True, batch_size: int = 10, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_types = num_types
        self.num_tokens = num_tokens
        self.num_subtrees = num_subtrees
        self.num_languages = num_languages

        self.include_token = include_token
        self.num_conv = num_conv
        self.conv_output_dim = conv_output_dim
        self.node_type_dim = node_type_dim
        self.node_token_dim = node_token_dim

        # node_dim is equal to conv_output_dim and subtree_dim
        self.node_dim = conv_output_dim
        self.subtree_dim = conv_output_dim

        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        language_matrix_index = [x for x in range(self.num_languages)]
        self.language_embeddings = tf.one_hot(language_matrix_index, self.num_languages)

        self.node_type_embeddings = None
        self.node_token_embeddings = None

        # init layers
        self.dense1 = tf.keras.layers.Dense(units=self.node_dim, activation=None, use_bias=False, name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=self.node_dim, activation=None, use_bias=False, name='dense2')
        self.tb_conv_layer = TBConvLayer(self.node_dim, self.conv_output_dim, self.num_conv, self.dropout_rate)
        self.aggregation_layer = AggregationLayer(name='code_encoder')

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def get_config(self):
        return {
            'num_types': self.num_types,
            'num_tokens': self.num_tokens,
            'num_subtrees': self.num_subtrees,
            'num_languages': self.num_languages,
            'include_token': self.include_token,
            'num_conv': self.num_conv,
            'conv_output_dim': self.conv_output_dim,
            'node_type_dim': self.node_type_dim,
            'node_token_dim': self.node_token_dim,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate
        }

    def build(self, input_shape):
        variance_scaling_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                             distribution="uniform")
        self.node_type_embeddings = self.add_weight('node_type_embeddings', shape=(self.num_types, self.node_type_dim),
                                                    initializer=variance_scaling_initializer)
        self.node_token_embeddings = self.add_weight('node_token_embeddings',
                                                     shape=(self.num_tokens, self.node_type_dim),
                                                     initializer=variance_scaling_initializer)

    def call(self, inputs):
        # extract inputs, shape = (batch_size, (...))
        language_index, node_type, node_tokens, children_index, children_node_tokens = inputs
        batch_size = tf.shape(children_node_tokens)[0]
        max_tree_size = tf.shape(children_node_tokens)[1]
        max_children = tf.shape(children_node_tokens)[2]

        # shape = (batch_size, max_tree_size, node_type_dim)
        # Example with batch size = 12: shape = (12, 48, 30)
        parent_node_type_embeddings = self._compute_parent_node_types_tensor(node_type)

        # shape = (batch_size, max_tree_size, max_children, node_type_dim)
        # Example with batch size = 12: shape = (12, 48, 8, 30)
        children_node_type_embeddings = self.tb_conv_layer.compute_children_node_types_tensor(
            parent_node_type_embeddings, children_index)

        if self.include_token:
            # shape = (batch_size, max_tree_size, node_token_dim)
            # Example with batch size = 12: shape = (12, 48, 50))
            parent_node_token_embeddings = self._compute_parent_node_tokens_tensor(node_tokens)

            # shape = (batch_size, max_tree_size, max_children, node_token_dim)
            # Example with batch size = 12: shape = (12, 48, 7, 50)
            children_node_token_embeddings = self._compute_children_node_tokens_tensor(children_node_tokens)

            # shape = (batch_size, max_tree_size, (node_type_dim + node_token_dim))
            # Example with batch size = 12: shape = (12, 48, (30 + 50))) = (12, 48, 80)
            parent_node_embeddings = tf.concat([parent_node_type_embeddings, parent_node_token_embeddings], -1)

            # shape = (batch_size, max_tree_size, max_children, (node_type_dim + node_token_dim))
            # Example with batch size = 12: shape = (12, 48, 6, (30 + 50))) = (12, 48, 6, 80)
            children_embeddings = tf.concat(
                [children_node_type_embeddings, children_node_token_embeddings], -1)

        else:
            self.logger.info("Excluding token information..........")
            # Example with batch size = 12: shape = (12, 48, (30 + 50))) = (12, 48, 80)
            parent_node_embeddings = parent_node_type_embeddings
            children_embeddings = children_node_type_embeddings

        language_tensor = tf.nn.embedding_lookup(params=self.language_embeddings, ids=language_index)

        language_tensor_for_parent = tf.reshape(language_tensor, [batch_size, 1, self.num_languages])
        language_tensor_for_parent = tf.tile(language_tensor_for_parent, [1, max_tree_size, 1])
        parent_node_embeddings = tf.concat([parent_node_embeddings, language_tensor_for_parent], 2)
        parent_node_embeddings = self.dense1(parent_node_embeddings)

        language_tensor_for_children = tf.reshape(language_tensor, [batch_size, 1, 1, self.num_languages])
        language_tensor_for_children = tf.tile(language_tensor_for_children, [1, max_tree_size, max_children, 1])
        children_embeddings = tf.concat([children_embeddings, language_tensor_for_children], -1)
        children_embeddings = self.dense2(children_embeddings)

        """Tree based Convolutional Layer"""
        # Example with batch size = 12 and num_conv = 8: shape = (12, 48, 128, 8)
        # Example with batch size = 1 and num_conv = 8: shape = (1, 48, 128, 8)
        conv_output = self.tb_conv_layer([parent_node_embeddings, children_embeddings, children_index])
        code_vector = self.aggregation_layer(conv_output)
        return code_vector

    def _compute_parent_node_types_tensor(self, parent_node_types_indices):
        parent_node_types_tensor = tf.nn.embedding_lookup(params=self.node_type_embeddings,
                                                          ids=parent_node_types_indices)
        return parent_node_types_tensor

    def _compute_parent_node_tokens_tensor(self, parent_node_tokens_indices):
        parent_node_tokens_tensor = tf.nn.embedding_lookup(params=self.node_token_embeddings,
                                                           ids=parent_node_tokens_indices)
        parent_node_tokens_tensor = tf.reduce_sum(input_tensor=parent_node_tokens_tensor, axis=2)
        return parent_node_tokens_tensor

    def _compute_children_node_tokens_tensor(self, children_node_tokens_indices):
        zero_vecs = tf.zeros((1, self.node_token_dim))
        vector_lookup = tf.concat([zero_vecs, self.node_token_embeddings[1:, :]], axis=0)
        # FIXME embedding_lookup OOM
        children_node_tokens_tensor = tf.nn.embedding_lookup(params=vector_lookup, ids=children_node_tokens_indices)
        children_node_tokens_tensor = tf.reduce_sum(input_tensor=children_node_tokens_tensor, axis=3)
        return children_node_tokens_tensor
