from pathlib import Path

# Path to the root of the project
BASE_DIR = Path(__file__).resolve().parent.parent
PACKAGE_DIR = BASE_DIR / 'infercode'
DATA_DIR = BASE_DIR / 'data'

LOG_DIR = DATA_DIR / 'logs'

DEBUG = False

# neural network
NODE_TYPE_DIM: int = 100
NODE_TOKEN_DIM: int = 100
CONV_OUTPUT_DIM: int = 100  # 50 100 200
NUM_CONV: int = 1
INCLUDE_TOKEN: bool = True
LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 10

# training params
EPOCHS = 40
CHECKPOINT_EVERY = 100  # Not used

# training workers
WORKERS: int = 4

# resource
PRETRAINED_MODEL_URL = "https://github.com/Aelita-S/infercode/releases/download/v0.0.1/model_saved.zip"
MODEL_NAME = "model"
NODE_TYPE_VOCAB_MODEL_PATH = "node_types/node_types_all.model"
NODE_TOKEN_VOCAB_MODEL_PATH = "tokens/universal_token_subword.model"
SUBTREE_VOCAB_MODEL_PATH = "subtrees/universal_subtree.model"

DATASET_DIR = DATA_DIR / 'datasets'
MODEL_DIR = DATA_DIR / 'models'
SAVED_MODEL_PATH = MODEL_DIR / f'{MODEL_NAME}_saved'
MODEL_CKPT_PATH = MODEL_DIR / f'{MODEL_NAME}_ckpt'

if __name__ == '__main__':
    print(BASE_DIR)
