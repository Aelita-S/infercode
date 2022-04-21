import logging
from pathlib import Path
from typing import Collection, Optional, Union

import numpy as np
import tensorflow as tf

from infercode.client.base_client import BaseClient


class InferCodeClient(BaseClient):
    logger = logging.getLogger('InferCodeTrainer')

    def __init__(self, language, *, model_path: Optional[Path] = None):
        super().__init__()
        self.language = language
        self.infercode_model = tf.keras.models.load_model(model_path or self.model_path, compile=False)

    def snippets_to_tensors(self, batch_code_snippets: Collection[Union[str, bytes]]):
        batch_tree_indexes = []
        assert len(batch_code_snippets) <= 5, "Batch size should be less than 5"
        for code_snippet in batch_code_snippets:
            # tree-sitter parser requires bytes as the input, not string
            if isinstance(code_snippet, str):
                code_snippet = code_snippet.encode(errors='ignore')
            ast = self.ast_parser.parse_with_language(code_snippet, self.language)
            tree_representation, _ = self.ast_util.simplify_ast(ast, code_snippet)
            tree_indexes = self.tensor_util.transform_tree_to_index(tree_representation)
            tree_indexes.update({'language_index': self.language_util.get_language_index(self.language)})
            batch_tree_indexes.append(tree_indexes)

        tensors, _ = self.tensor_util.trees_to_batch_tensors(batch_tree_indexes)
        return tensors

    def encode(self, batch_code_snippets: Collection[Union[str, bytes]]) -> np.ndarray:
        tensors = self.snippets_to_tensors(batch_code_snippets)
        return self.infercode_model.predict(tensors)
