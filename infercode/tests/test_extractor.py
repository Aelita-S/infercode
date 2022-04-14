# from utils.token_vocab_extractor import TokenVocabExtractor
# extractor = TokenVocabExtractor("OJ_raw_small", "OJ_raw_word", "word")
# extractor.create_vocab()
from infercode.data_utils.subtree_vocab_extractor import SubtreeVocabExtractor

# node_type_vocab_model_path= "sentencepiece_vocab/node_types/node_types_c"
# node_token_vocab_model_path= "sentencepiece_vocab/tokens/OJ_raw_bpe"
# extractor = ASTUtil(node_type_vocab_model_path, node_token_vocab_model_path, "c")
# text = open("test.c", "rb").read()
# subtrees = extractor.extract_subtrees(text)
# for s in subtrees:
#     print(s)
data_path = "OJ_raw_small"
node_type_vocab_path = "sentencepiece_vocab/node_types/node_types_c.model"
node_token_vocab_path = "sentencepiece_vocab/tokens/OJ_2.model"

# extractor = TokenVocabExtractor(data_path, node_token_vocab_prefix)
# extractor.create_vocab()
output = "sentencepiece_vocab/subtrees/OJ_2"
subtree_vocab_extractor = SubtreeVocabExtractor(data_path, output, node_type_vocab_path,
                                                node_token_vocab_path, "c")
subtree_vocab_extractor.create_vocab()
