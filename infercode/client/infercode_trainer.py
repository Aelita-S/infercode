import logging
from pathlib import Path

import tensorflow as tf

from infercode.data_utils.dataset_processor import DatasetProcessor
from infercode.data_utils.sequence import DataSequence
from infercode.network.infercode_network import InferCodeModel
from .base_client import BaseClient
from ..network.loss import SampledSoftmaxLoss
from ..settings import LOG_DIR


class InferCodeTrainer(BaseClient):
    logger = logging.getLogger('InferCodeTrainer')

    def __init__(self):
        super().__init__()
        self.train_data_seq = None
        self.val_data_seq = None
        # ------------Set up the neural network------------
        self.model = self._build_model()

    def _build_model(self):
        # input layers (excluding the batch dimension)
        language_index = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='language_index')
        node_type = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='node_type')
        node_tokens = tf.keras.layers.Input(shape=(None, 1), dtype=tf.int32, name='node_tokens')
        children_index = tf.keras.layers.Input(shape=(None, None), dtype=tf.int32, name='children_index')
        children_node_tokens = tf.keras.layers.Input(shape=(None, None, 1), dtype=tf.int32,
                                                     name='children_node_tokens')
        num_subtrees = self.subtree_vocab.get_vocabulary_size()
        infercode_model = InferCodeModel(num_types=self.node_type_vocab.get_vocabulary_size(),
                                         num_tokens=self.node_token_vocab.get_vocabulary_size(),
                                         num_subtrees=num_subtrees,
                                         num_languages=self.language_util.get_num_languages(),
                                         num_conv=self.num_conv,
                                         node_type_dim=self.node_type_dim,
                                         node_token_dim=self.node_token_dim,
                                         conv_output_dim=self.conv_output_dim,
                                         include_token=self.include_token,
                                         batch_size=self.batch_size,
                                         dropout_rate=0.4)
        code_vector = infercode_model([language_index, node_type, node_tokens, children_index, children_node_tokens])
        model = tf.keras.Model(inputs=[language_index, node_type, node_tokens, children_index, children_node_tokens],
                               outputs=code_vector)
        # Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer, loss=SampledSoftmaxLoss(num_subtrees, self.conv_output_dim))
        return model

    def process_data_sequence(self, input_data_path: Path, output_processed_data_path: Path,
                              val_data_path: Path = None, val_processed_data_path: Path = None):
        training_data_processor = DatasetProcessor(input_data_path=input_data_path,
                                                   output_tensors_path=output_processed_data_path,
                                                   node_type_vocab_model_path=self.node_type_vocab_model_path,
                                                   node_token_vocab_model_path=self.node_token_vocab_model_path,
                                                   subtree_vocab_model_path=self.subtree_vocab_model_path)
        training_buckets = training_data_processor.process_or_load_data()
        self.train_data_seq = DataSequence(self.batch_size, training_buckets)

        if val_data_path and val_processed_data_path:
            val_data_processor = DatasetProcessor(input_data_path=val_data_path,
                                                  output_tensors_path=val_processed_data_path,
                                                  node_type_vocab_model_path=self.node_type_vocab_model_path,
                                                  node_token_vocab_model_path=self.node_token_vocab_model_path,
                                                  subtree_vocab_model_path=self.subtree_vocab_model_path)
            validation_buckets = val_data_processor.process_or_load_data()
            self.val_data_seq = DataSequence(self.batch_size, validation_buckets)

    def train(self, workers: int = 1):
        """

        :param workers: Use multiprocessing. Recommended to use 2-4 workers(will increase memory usage).
        :return: Model training history
        """
        assert self.train_data_seq is not None, "Please process training data first. Use process_data_sequence()"
        # monitor training loss, if it is not improving, stop training
        loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path / 'ckpt',
            monitor='val_loss',
            save_best_only=True)

        # use_multiprocessing=True while using multiple workers
        history = self.model.fit(self.train_data_seq, epochs=self.epochs,
                                 callbacks=[loss_callback, tensorboard_callback, model_checkpoint_callback],
                                 batch_size=self.batch_size,
                                 validation_data=self.val_data_seq,
                                 workers=workers,
                                 use_multiprocessing=workers > 1)
        self.model.summary(expand_nested=True)
        self.model.save(self.model_path)
        return history

    def test_train(self):
        self.model.train_on_batch(x=self.train_data_seq[0][0], y=self.train_data_seq[0][1])
        self.model.summary(expand_nested=True)
        self.model.save(self.model_path)
