from pathlib import Path

from tree_sitter import Language, Parser


class ASTParser:
    import logging
    logger = logging.getLogger('ASTParser')

    def __init__(self, language=None):
        # ------------ To initialize for the treesitter parser ------------
        p = Path.home() / '.tree-sitter' / 'bin'
        self.Languages = {}
        for file in p.glob("*.so"):
            lang = file.name.split('.')[0]
            self.Languages[lang] = Language(str(file), lang)
        self.parser = Parser()

        self.language = language
        if self.language is None:
            self.logger.info(
                "Cannot find language configuration, using java parser as the default to parse the code into AST")
            self.language = "java"

        lang = self.Languages.get(self.language)
        self.parser.set_language(lang)
        # -----------------------------------------------------------------

    def parse_with_language(self, code_snippet, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
        return self.parser.parse(code_snippet)

    def parse(self, code_snippet):
        return self.parser.parse(code_snippet)

    def set_language(self, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
