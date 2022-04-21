import tensorflow as tf

variance_scaling_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")


class TBConvLayer(tf.keras.layers.Layer):
    """Tree-based Convolutional and Pooling Layer."""

    def __init__(self, feature_size: int, output_size: int, num_conv: int, dropout_rate: float, *args, **kwargs):
        """

        :param feature_size: The size of the feature vector
        :param output_size: The size of the output vector
        :param num_conv: The number of convolutions to perform
        """
        super(TBConvLayer, self).__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_conv = num_conv
        self.dropout_rate = dropout_rate

        self.w_r = None
        self.w_l = None
        self.w_t = None
        self.b_conv = None

    def build(self, input_shape):
        self.w_t = self.add_weight("w_t", shape=(self.num_conv, self.feature_size, self.output_size),
                                   initializer=variance_scaling_initializer)
        self.w_l = self.add_weight("w_l", shape=(self.num_conv, self.feature_size, self.output_size),
                                   initializer=variance_scaling_initializer)
        self.w_r = self.add_weight("w_r", shape=(self.num_conv, self.feature_size, self.output_size),
                                   initializer=variance_scaling_initializer)
        self.b_conv = self.add_weight("b_conv", shape=(self.num_conv, self.output_size,),
                                      initializer=variance_scaling_initializer)

    def get_config(self):
        return {
            "feature_size": self.feature_size,
            "output_size": self.output_size,
            "num_conv": self.num_conv
        }

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        """Creates a convolution layer with num_conv convolutions merged together at
        the output. Final ou tput will be a tensor with shape
        [batch_size, num_nodes, output_size * num_conv]"""
        parent_embeddings, children_embeddings, children_indices = inputs
        for layer in range(self.num_conv):
            parent_embeddings = self.conv_step(parent_embeddings, children_embeddings, children_indices, layer)
            if training:
                parent_embeddings = tf.nn.dropout(parent_embeddings, rate=self.dropout_rate)  # DROP-OUT here
            children_embeddings = self.compute_children_node_types_tensor(parent_embeddings, children_indices)
        return parent_embeddings

    def conv_step(self, parent_embeddings, children_embeddings, children_indices, layer):
        """Convolve a batch of nodes and children.

        Lots of high dimensional tensors in this function. Intuitively it makes
        more sense if we did this work with while loops, but computationally this
        is more efficient. Don't try to wrap your head around all the tensor dot
        products, just follow the trail of dimensions.
        """
        with tf.name_scope('conv_step'):
            # nodes is shape (batch_size x max_tree_size x feature_size)
            # children is shape (batch_size x max_tree_size x max_children)

            with tf.name_scope('trees'):
                # add a 4th dimension to the nodes tensor
                parent_embeddings = tf.expand_dims(parent_embeddings, axis=2)
                # tree_tensor is shape
                # (batch_size x max_tree_size x max_children + 1 x feature_size)
                tree_tensor = tf.concat([parent_embeddings, children_embeddings], axis=2, name='trees')

            with tf.name_scope('coefficients'):
                # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
                c_t = self.eta_t(children_indices)
                c_r = self.eta_r(children_indices, c_t)
                c_l = self.eta_l(children_indices, c_t, c_r)

                # concatenate the position coefficients into a tensor
                # (batch_size x max_tree_size x max_children + 1 x 3)
                coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

            with tf.name_scope('weights'):
                # stack weight matrices on top to make a weight tensor
                # (3, feature_size, output_size)
                weights = tf.stack([self.w_t[layer], self.w_r[layer], self.w_l[layer]], axis=0)

            with tf.name_scope('combine'):
                batch_size = tf.shape(children_indices)[0]
                max_tree_size = tf.shape(children_indices)[1]
                max_children = tf.shape(children_indices)[2]

                # reshape for matrix multiplication
                x = batch_size * max_tree_size
                y = max_children + 1
                result = tf.reshape(tree_tensor, (x, y, self.feature_size))
                coef = tf.reshape(coef, (x, y, 3))
                result = tf.matmul(result, coef, transpose_a=True)
                result = tf.reshape(result, (batch_size, max_tree_size, 3, self.feature_size))

                # output is (batch_size, max_tree_size, output_size)
                result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

                output = tf.nn.leaky_relu(result + self.b_conv[layer])
                # output = tf.compat.v1.nn.swish(result + b_conv)
                # output = tf.layers.batch_normalization(output, training=self.placeholders['is_training'])
                return output

    def compute_children_node_types_tensor(self, parent_embeddings, children_indices):
        """Build the children tensor from the input nodes and child lookup."""
        with tf.name_scope('children_tensor'):
            max_children = tf.shape(children_indices)[2]
            batch_size = tf.shape(parent_embeddings)[0]
            num_nodes = tf.shape(parent_embeddings)[1]

            # replace the root node with the zero vector so lookups for the 0th
            # vector return 0 instead of the root vector
            # zero_vecs is (batch_size, num_nodes, 1)
            zero_vecs = tf.zeros((batch_size, 1, self.feature_size))
            # vector_lookup is (batch_size x num_nodes x feature_size)
            vector_lookup = tf.concat([zero_vecs, parent_embeddings[:, 1:, :]], axis=1)
            # children is (batch_size x num_nodes x num_children x 1)
            children_indices = tf.expand_dims(children_indices, axis=3)
            # prepend the batch indices to the 4th dimension of children
            # batch_indices is (batch_size x 1 x 1 x 1)
            batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
            # batch_indices is (batch_size x num_nodes x num_children x 1)
            batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
            # children is (batch_size x num_nodes x num_children x 2)
            children_indices = tf.concat([batch_indices, children_indices], axis=3)
            # output will have shape (batch_size x num_nodes x num_children x feature_size)
            # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
            return tf.gather_nd(vector_lookup, children_indices, name='children')

    def eta_t(self, children_indices):
        """Compute weight matrix for how much each vector belongs to the 'top'"""
        with tf.name_scope('coef_t'):
            # children is shape (batch_size x max_tree_size x max_children)
            batch_size = tf.shape(children_indices)[0]
            max_tree_size = tf.shape(children_indices)[1]
            max_children = tf.shape(children_indices)[2]
            # eta_t is shape (batch_size x max_tree_size x max_children + 1)
            return tf.tile(tf.expand_dims(tf.concat(
                [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
                axis=1), axis=0,
            ), [batch_size, 1, 1], name='coef_t')

    def eta_r(self, children_indices, t_coef):
        """Compute weight matrix for how much each vector belogs to the 'right'"""
        with tf.name_scope('coef_r'):
            # children is shape (batch_size x max_tree_size x max_children)
            children_indices = tf.cast(children_indices, tf.float32)
            batch_size = tf.shape(children_indices)[0]
            max_tree_size = tf.shape(children_indices)[1]
            max_children = tf.shape(children_indices)[2]

            # num_siblings is shape (batch_size x max_tree_size x 1)
            num_siblings = tf.cast(
                tf.math.count_nonzero(children_indices, axis=2, keepdims=True),
                dtype=tf.float32
            )
            # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
            num_siblings = tf.tile(
                num_siblings, [1, 1, max_children + 1], name='num_siblings'
            )
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.minimum(children_indices, tf.ones(tf.shape(input=children_indices)))],
                axis=2, name='mask'
            )

            # child indices for every tree (batch_size x max_tree_size x max_children + 1)
            child_indices = tf.multiply(tf.tile(
                tf.expand_dims(
                    tf.expand_dims(
                        tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                        axis=0
                    ),
                    axis=0
                ),
                [batch_size, max_tree_size, 1]
            ), mask, name='child_indices')

            # weights for every tree node in the case that num_siblings = 0
            # shape is (batch_size x max_tree_size x max_children + 1)
            singles = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.fill((batch_size, max_tree_size, 1), 0.5),
                 tf.zeros((batch_size, max_tree_size, max_children - 1))],
                axis=2, name='singles')

            # eta_r is shape (batch_size x max_tree_size x max_children + 1)
            return tf.where(
                condition=tf.equal(num_siblings, 1.0),
                # avoid division by 0 when num_siblings == 1
                x=singles,
                # the normal case where num_siblings != 1
                y=tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
                name='coef_r'
            )

    def eta_l(self, children_indices, coef_t, coef_r):
        """Compute weight matrix for how much each vector belongs to the 'left'"""
        with tf.name_scope('coef_l'):
            children_indices = tf.cast(children_indices, tf.float32)
            batch_size = tf.shape(children_indices)[0]
            max_tree_size = tf.shape(children_indices)[1]
            # creates a mask of 1's and 0's where 1 means there is a child there
            # has shape (batch_size x max_tree_size x max_children + 1)
            mask = tf.concat(
                [tf.zeros((batch_size, max_tree_size, 1)),
                 tf.minimum(children_indices, tf.ones(tf.shape(input=children_indices)))],
                axis=2,
                name='mask'
            )

            # eta_l is shape (batch_size x max_tree_size x max_children + 1)
            return tf.multiply(
                tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
            )


class AggregationLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(AggregationLayer, self).__init__(*args, **kwargs)
        self.W = None
        self.node_dim = None

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def build(self, input_shape):
        self.node_dim = input_shape[-1]
        self.W = self.add_weight('w_attention', shape=(input_shape[-1], 1), initializer=variance_scaling_initializer)

    def call(self, inputs, **kwargs):
        """
        :param inputs: (batch_size * max_graph_size, self.node_dim)
        :return:
        """
        max_tree_size = tf.shape(input=inputs)[1]

        flat_nodes_representation = tf.reshape(inputs, [-1, self.node_dim])
        aggregated_vector = tf.matmul(flat_nodes_representation, self.W)

        attention_score = tf.reshape(aggregated_vector, [-1, max_tree_size, 1])

        """A note here: softmax will distributed the weights to all of the nodes (sum of node weghts = 1),
        an interesting finding is that for some nodes, the attention score will be very very small, i.e e-12, 
        thus making parts of aggregated vector becomes near zero and affect on the learning (very slow nodes_representationergence
        - Better to use sigmoid"""

        attention_weights = tf.nn.softmax(attention_score, axis=1)

        # attention_weights = tf.nn.sigmoid(attention_score)

        # TODO: reduce_max vs reduce_sum vs reduce_mean
        # if aggregation_type == 1:
        #     print("Using tf.reduce_sum...........")
        weighted_average_nodes = tf.reduce_sum(input_tensor=tf.multiply(inputs, attention_weights), axis=1)
        # if aggregation_type == 2:
        # print("Using tf.reduce_max...........")
        # weighted_average_nodes = tf.reduce_max(tf.multiply(nodes_representation, attention_weights), axis=1)
        # if aggregation_type == 3:
        # print("Using tf.reduce_mean...........")
        # weighted_average_nodes = tf.reduce_mean(tf.multiply(nodes_representation, attention_weights), axis=1)

        return weighted_average_nodes


class SampledSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, num_sampled: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_sampled = num_sampled
        self.dim = None
        self.W = None
        self.B = None

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def build(self, input_shape):
        self.dim = input_shape[-1][-1]
        self.W = self.add_weight('loss_weights', shape=(self.num_classes, self.dim),
                                 initializer=variance_scaling_initializer)
        self.B = self.add_weight('loss_biases', shape=(self.num_classes,),
                                 initializer=variance_scaling_initializer)

    def call(self, inputs, training=None, **kwargs):
        y_true, y_pred = inputs
        sampled_softmax_loss = tf.nn.sampled_softmax_loss(weights=self.W,
                                                          biases=self.B,
                                                          labels=y_true,
                                                          inputs=y_pred,
                                                          num_sampled=self.num_sampled,
                                                          num_classes=self.num_classes)
        loss = tf.reduce_mean(input_tensor=sampled_softmax_loss)
        return loss
