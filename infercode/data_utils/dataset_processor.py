import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm

from .ast_parser import ASTParser
from .ast_util import ASTUtil
from .language_util import LanguageUtil
from .subtree_util import SubtreeUtil
from .subtree_vocab_extractor import SubtreeVocabExtractor
from .tensor_util import TensorUtil
from .token_vocab_extractor import TokenVocabExtractor
from .vocabulary import Vocabulary


class DatasetProcessor:
    logger = logging.getLogger('DatasetProcessor')

    def __init__(self, input_data_path: Union[str, Path],
                 output_tensors_path: Union[str, Path],
                 node_type_vocab_model_path: Union[str, Path],
                 node_token_vocab_model_path: Union[str, Path],
                 subtree_vocab_model_path: Union[str, Path]):

        self.input_data_path = input_data_path
        self.output_tensors_path = output_tensors_path
        self.node_type_vocab_model_path = node_type_vocab_model_path
        self.node_token_vocab_model_path = node_token_vocab_model_path
        self.subtree_vocab_model_path = subtree_vocab_model_path

        self.ast_parser = ASTParser()
        self.subtree_util = SubtreeUtil()
        self.language_util = LanguageUtil()

        self.token_vocab_extractor = TokenVocabExtractor(
            node_token_vocab_model_path=self.node_token_vocab_model_path,
            model_type="bpe")
        self.subtree_vocab_extractor = SubtreeVocabExtractor(subtree_vocab_model_path=self.subtree_vocab_model_path)

        self.init_vocabs()
        self.tensor_util = TensorUtil()

        # AST Util can only be initialized after extracted the token vocab
        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_path,
                                node_token_vocab_model_path=self.node_token_vocab_model_path)

    def detect_language_of_file(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        return self.language_util.get_language_by_file_extension(file_extension)

    def put_trees_into_buckets(self):
        """Trees with similar size should be put into the same bucket"""
        bucket_sizes = np.array(list(range(20, 7500, 20)))
        buckets = defaultdict(list)

        for i, (subdir, dirs, files) in enumerate(os.walk(self.input_data_path)):
            for file in tqdm(files, desc=f"Processing dir {subdir}: "):
                language = self.detect_language_of_file(file)
                language_index = self.language_util.get_language_index(language)
                file_path = os.path.join(subdir, file)
                with open(file_path, "rb") as f:
                    code_snippet = f.read()
                ast = self.ast_parser.parse_with_language(code_snippet, language=language)
                tree_representation, tree_size = self.ast_util.simplify_ast(ast, code_snippet.decode('utf-8', 'ignore'))

                tree_indexes = self.tensor_util.transform_tree_to_index(tree_representation)
                tree_indexes["language_index"] = language_index
                tree_indexes["size"] = tree_size

                # Extract all subtrees from the code snippet
                subtrees = self.subtree_util.extract_subtrees(ast)

                # ----------Convert subtree string to id----------
                subtrees_id = []
                for subtree in subtrees:
                    if len(subtree) >= 4 and len(subtree) <= 16:
                        subtree_str = "-".join(subtree)
                        subtree_id = self.subtree_vocab.get_id_from_piece(subtree_str)
                        if subtree_id != 0:
                            subtrees_id.append(subtree_id)

                # Assert to make sure the list of subtrees must NOT be 0, if it is 0, then it's likely a bug
                # assert len(subtrees_id) > 0
                if len(subtrees_id) > 0:
                    subtrees_id = list(set(subtrees_id))

                    # Put different instances of the same snippet (with different subtree id) into buckets for training
                    for subtree_id in subtrees_id:
                        tree_indexes["subtree_id"] = subtree_id
                        chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
                        buckets[chosen_bucket_idx].append(tree_indexes)

        self.logger.info("Saving processed data into pickle format.....")
        pickle.dump(buckets, open(self.output_tensors_path, "wb"))

        return buckets

    def init_vocabs(self):
        if not os.path.exists(self.node_token_vocab_model_path):
            self.logger.info("Generating token vocabulary")
            self.token_vocab = self.token_vocab_extractor.create_vocab_from_dir(self.input_data_path)
        else:
            self.token_vocab = Vocabulary(100000, self.node_token_vocab_model_path)

        if not os.path.exists(self.subtree_vocab_model_path):
            self.logger.info("Generating subtree vocabulary")
            self.subtree_vocab = self.subtree_vocab_extractor.create_vocab_from_dir(self.input_data_path)
        else:
            self.subtree_vocab = Vocabulary(100000, self.subtree_vocab_model_path)

    def process_or_load_data(self):
        if not os.path.exists(self.output_tensors_path):
            self.logger.info("Processing the dataset")
            training_buckets = self.put_trees_into_buckets()
        else:
            training_buckets = pickle.load(open(self.output_tensors_path, "rb"))

        return training_buckets
