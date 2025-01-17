import os

# Change from -1 to 0 to enable GPU
from infercode.settings import SAVED_MODEL_PATH

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import logging

logging.getLogger('tensorflow').disabled = True
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
from infercode.client.infercode_client import InferCodeClient

if __name__ == '__main__':
    infercode = InferCodeClient(model_path=SAVED_MODEL_PATH)
    infercode_remote = InferCodeClient(remote_model_predict_url='http://127.0.0.1:8501/v1/models/model:predict')
    vectors = infercode.encode('c', ["for (i = 0; i < n; i++)", "struct book{ int num; char s[27]; }shu[1000];"])
    print(vectors)
    vectors = infercode_remote.encode('c', ["for (i = 0; i < n; i++)", "struct book{ int num; char s[27]; }shu[1000];"])
    print(vectors)
