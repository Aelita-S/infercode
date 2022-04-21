import logging
import urllib.request

from tqdm import tqdm

from infercode.data_utils.ast_parser import ASTParser
from infercode.data_utils.ast_util import ASTUtil
from infercode.data_utils.language_util import LanguageUtil
from infercode.data_utils.tensor_util import TensorUtil
from infercode.data_utils.vocabulary import Vocabulary
from infercode.settings import *


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

    def __init__(self):
        # Training params
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.checkpoint_every = CHECKPOINT_EVERY

        self.num_conv = NUM_CONV
        self.node_type_dim = NODE_TYPE_DIM
        self.node_token_dim = NODE_TOKEN_DIM
        self.conv_output_dim = CONV_OUTPUT_DIM
        self.include_token = INCLUDE_TOKEN
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE

        self.model_name = MODEL_NAME
        self.pretrained_model_url = PRETRAINED_MODEL_URL

        self.model_path = SAVED_MODEL_PATH
        self.workers = WORKERS

        # Init vocab
        self.node_type_vocab_model_path = PACKAGE_DIR / "sentencepiece_vocab" / NODE_TYPE_VOCAB_MODEL_PATH
        self.node_token_vocab_model_path = PACKAGE_DIR / "sentencepiece_vocab" / NODE_TOKEN_VOCAB_MODEL_PATH
        self.subtree_vocab_model_path = PACKAGE_DIR / "sentencepiece_vocab" / SUBTREE_VOCAB_MODEL_PATH

        self.node_type_vocab = Vocabulary(100000, self.node_type_vocab_model_path)
        self.node_token_vocab = Vocabulary(100000, self.node_token_vocab_model_path)
        self.subtree_vocab = Vocabulary(100000, self.subtree_vocab_model_path)

        self.ast_parser = ASTParser()
        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_path,
                                node_token_vocab_model_path=self.node_token_vocab_model_path)

        self.language_util = LanguageUtil()

        self.tensor_util = TensorUtil()
