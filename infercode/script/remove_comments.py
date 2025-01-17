import re

# first group captures quoted strings (double or single)
# second group captures comments (//single-line or /* multi-line */)
pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
regex = re.compile(pattern, re.MULTILINE | re.DOTALL)


def remove_comments(string):
    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, string)
