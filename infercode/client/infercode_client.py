import gzip
import json
import logging
from pathlib import Path
from typing import Collection, Optional, Union

import numpy as np
import requests
import tensorflow as tf

from infercode.client.base_client import BaseClient


class InferCodeClient(BaseClient):
    logger = logging.getLogger('InferCodeTrainer')

    def __init__(self, *, model_path: Optional[Path] = None, remote_predict_url: Optional[str] = None):
        """

        :param model_path:
        :param remote_predict_url: tf serving url, such as http://127.0.0.1:8501/v1/models/model:predict
        """
        super().__init__()
        self.model_path = model_path
        self.use_remote_model = remote_predict_url is not None
        assert model_path is not None or remote_predict_url is not None, \
            "model_path or remote_model_predict_url should be provided"
        if model_path:
            self.infercode_model = tf.keras.models.load_model(model_path or self.model_path, compile=False)
        self.remote_model_predict_url = remote_predict_url

    def snippets_to_tensors(self, batch_code_snippets: Collection[Union[str, bytes]], language: str = 'c') -> list:
        batch_tree_indexes = []
        assert len(batch_code_snippets) <= 5, "Batch size should be less than 5"
        for code_snippet in batch_code_snippets:
            # tree-sitter parser requires bytes as the input, not string
            if isinstance(code_snippet, str):
                code_snippet = code_snippet.encode(errors='ignore')
            ast = self.ast_parser.parse_with_language(code_snippet, language)
            tree_representation, _ = self.ast_util.simplify_ast(ast, code_snippet)
            tree_indexes = self.tensor_util.transform_tree_to_index(tree_representation)
            tree_indexes.update({'language_index': self.language_util.get_language_index(language)})
            batch_tree_indexes.append(tree_indexes)
        tensors, _ = self.tensor_util.trees_to_batch_tensors(batch_tree_indexes,
                                                             return_np_array=not self.use_remote_model)
        return tensors

    def _remote_predict(self, tensors: list[dict]) -> np.ndarray:
        batch_size = len(tensors[0])
        req_body = {
            'instances': [{
                'language_index': tensors[0][i],
                'node_type': tensors[1][i],
                'node_tokens': tensors[2][i],
                'children_index': tensors[3][i],
                'children_node_tokens': tensors[4][i],
            } for i in range(batch_size)]
        }
        gzip_req_body = gzip.compress(json.dumps(req_body).encode('utf-8'))
        response = requests.post(self.remote_model_predict_url, data=gzip_req_body,
                                 headers={
                                     'content-type': 'application/json',
                                     'content-encoding': 'gzip',
                                 }, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Remote model predict failed with status code {response.status_code}, message: \n{response.text}")
        predictions = response.json()['predictions']
        return np.asarray(predictions)

    def encode(self, batch_code_snippets: Collection[Union[str, bytes]], language: Optional[str] = 'c') -> np.ndarray:
        tensors = self.snippets_to_tensors(batch_code_snippets, language)
        if self.use_remote_model:
            return self._remote_predict(tensors)
        return self.infercode_model.predict(tensors)
