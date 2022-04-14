import configparser
import logging
import pathlib
import urllib.request
import zipfile
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from infercode.configs.__version__ import version
from infercode.data_utils.ast_parser import ASTParser
from infercode.data_utils.ast_util import ASTUtil
from infercode.data_utils.language_util import LanguageUtil
from infercode.data_utils.tensor_util import TensorUtil
from infercode.data_utils.vocabulary import Vocabulary

tf.compat.v1.disable_v2_behavior()
package_dir = Path(__file__).parents[1]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class BaseClient:
    logger = logging.getLogger('BaseClient')

    def __init__(self, language, config=None):
        self.language = language
        if config is None:
            current_path = pathlib.Path(__file__).parent
            parent_of_current_path = current_path.parent.absolute()
            config = configparser.ConfigParser()  # TODO use python file as config
            default_config_path = parent_of_current_path / "configs/default_config.ini"
            config.read(default_config_path)

        self.resource_config = config["resource"]
        self.training_config = config["training_params"]
        self.nn_config = config["neural_network"]

        # Training params
        self.epochs = int(self.training_config["epochs"])
        self.batch_size = int(self.nn_config["batch_size"])
        self.checkpoint_every = int(self.training_config["checkpoint_every"])

        self.num_conv = int(self.nn_config["num_conv"])
        self.node_type_dim = int(self.nn_config["node_type_dim"])
        self.node_token_dim = int(self.nn_config["node_token_dim"])
        self.conv_output_dim = int(self.nn_config["conv_output_dim"])
        self.include_token = int(self.nn_config["include_token"])
        self.batch_size = int(self.nn_config["batch_size"])
        self.learning_rate = float(self.nn_config["lr"])

        self.model_name = self.resource_config["model_name"]
        self.pretrained_model_url = self.resource_config["pretrained_model_url"]
        self.version = version

        # Init vocab
        self.node_type_vocab_model_path = package_dir / "sentencepiece_vocab" / self.resource_config[
            "node_type_vocab_model_path"]
        self.node_token_vocab_model_path = package_dir / "sentencepiece_vocab" / self.resource_config[
            "node_token_vocab_model_path"]
        self.subtree_vocab_model_path = package_dir / "sentencepiece_vocab" / self.resource_config[
            "subtree_vocab_model_path"]

        home = Path.home()
        model_checkpoint = home / ".infercode_data" / "model_checkpoint" / self.model_name
        model_checkpoint_ckpt = model_checkpoint / "cnn_tree.ckpt.index"

        if not model_checkpoint.exists():
            model_checkpoint.mkdir(parents=True, exist_ok=True)

        """
        Comment out this part if training locally
        """
        if not model_checkpoint_ckpt.exists():
            pretrained_model_checkpoint_target = model_checkpoint / "universal_model.zip"
            if not pretrained_model_checkpoint_target.exists():
                download_url(self.pretrained_model_url, pretrained_model_checkpoint_target)
            with zipfile.ZipFile(pretrained_model_checkpoint_target, 'r') as zip_ref:
                zip_ref.extractall(model_checkpoint)

        self.model_checkpoint = model_checkpoint
        self.node_type_vocab = Vocabulary(100000, self.node_type_vocab_model_path)
        self.node_token_vocab = Vocabulary(100000, self.node_token_vocab_model_path)
        self.subtree_vocab = Vocabulary(100000, self.subtree_vocab_model_path)

        self.ast_parser = ASTParser(self.language)
        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_path,
                                node_token_vocab_model_path=self.node_token_vocab_model_path)

        self.language_util = LanguageUtil()
        self.language_index = self.language_util.get_language_index(self.language)

        self.tensor_util = TensorUtil()
