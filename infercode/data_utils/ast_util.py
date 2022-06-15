import logging
from pathlib import Path
from typing import Optional, Union

import tree_sitter

from infercode.data_utils import identifiersplitting
from infercode.data_utils.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class ASTNode:
    def __init__(self, node_type: str, node_type_id: int, node_tokens: Optional[list[str]] = None,
                 node_tokens_id: Optional[list[int]] = None, children: Optional[list['ASTNode']] = None):
        self.node_type = node_type
        self.node_type_id = node_type_id
        self.node_tokens = node_tokens if node_tokens else []
        self.node_tokens_id = node_tokens_id if node_tokens_id else []
        self.children = children if children else []

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)


class ASTUtil:
    def __init__(self, node_type_vocab_model_path: Union[str, Path], node_token_vocab_model_path: Union[str, Path]):
        self.type_vocab = Vocabulary(1000, node_type_vocab_model_path)
        self.token_vocab = Vocabulary(100000, node_token_vocab_model_path)

    # Simplify the AST
    def simplify_ast(self, tree: tree_sitter.Tree, text: Union[str, bytes]):
        # tree = self.ast_parser.parse(text)
        # convert bytes to string, ignore decoding errors
        if isinstance(text, bytes):
            text = text.decode('utf-8', 'ignore')
        root: tree_sitter.Node = tree.root_node

        ignore_types = ["\n"]
        num_nodes = 0
        root_type = str(root.type)
        root_type_id = self.type_vocab.get_id_or_unk_for_text(root_type)[0]
        queue = [root]

        simplify_root = ASTNode(node_type=root_type, node_type_id=root_type_id)

        simplify_queue = [simplify_root]
        while queue:

            current_node = queue.pop(0)
            simplify_current_node = simplify_queue.pop(0)
            num_nodes += 1

            for child in current_node.children:
                child_type = str(child.type)
                if child_type in ignore_types:
                    continue
                queue.append(child)

                child_type_id = self.type_vocab.get_id_or_unk_for_text(child_type)[0]

                child_sub_tokens_id = []
                child_sub_tokens = []

                has_child = len(child.children) > 0

                if not has_child:
                    child_token = text[child.start_byte:child.end_byte]
                    child_sub_tokens_id = self.token_vocab.get_id_or_unk_for_text(child_token)
                    subtokens = " ".join(identifiersplitting.split_identifier_into_parts(child_token))
                    child_sub_tokens = self.token_vocab.tokenize(subtokens)

                if len(child_sub_tokens_id) == 0:
                    child_sub_tokens_id.append(0)
                else:
                    child_sub_tokens_id = [x for x in child_sub_tokens_id if x != 0]

                simplify_child = ASTNode(node_type=child_type, node_type_id=child_type_id, node_tokens=child_sub_tokens,
                                         node_tokens_id=child_sub_tokens_id)

                simplify_current_node.add_child(simplify_child)
                simplify_queue.append(simplify_child)
        return simplify_root, num_nodes


if __name__ == '__main__':
    node = ASTNode(node_type="a", node_type_id=1)
    print(node.children)
