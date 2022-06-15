import logging
import os

from infercode.client.infercode_client import InferCodeClient
from infercode.settings import SAVED_MODEL_PATH

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Change from -1 to 0 to enable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    infercode = InferCodeClient(model_path=SAVED_MODEL_PATH)
    vectors = infercode.encode(["for (i = 0; i < n; i++)", "struct book{ int num; char s[27]; }shu[1000];"])

    print(vectors)
