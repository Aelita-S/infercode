import tensorflow as tf

from infercode.network.layers import variance_scaling_initializer


class SampledSoftmaxLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, dim: int, num_sampled: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_sampled = num_sampled
        self.dim = dim
        self.W = tf.Variable(variance_scaling_initializer((self.num_classes, self.dim)), name='loss_weights')
        self.B = tf.Variable(variance_scaling_initializer((self.num_classes,)), name='loss_biases')

    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'dim': self.dim,
            'num_sampled': self.num_sampled,
        }

    def call(self, y_true, y_pred):
        sampled_softmax_loss = tf.nn.sampled_softmax_loss(weights=self.W,
                                                          biases=self.B,
                                                          labels=y_true,
                                                          inputs=y_pred,
                                                          num_sampled=self.num_sampled,
                                                          num_classes=self.num_classes)
        loss = tf.reduce_mean(input_tensor=sampled_softmax_loss)
        return loss
