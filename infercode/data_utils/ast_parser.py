import logging
from pathlib import Path

from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)


class ASTParser:

    def __init__(self):
        # ------------ To initialize for the treesitter parser ------------
        p = Path.home() / '.tree-sitter' / 'bin'
        self.Languages = {}
        for file in p.glob("*.so"):
            lang = file.name.split('.')[0]
            self.Languages[lang] = Language(str(file), lang)
        self.parser = Parser()

    def parse_with_language(self, code_snippet, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
        return self.parser.parse(code_snippet)
