import re
from lxml import etree

# Remove comments
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)

# Match floating point numbers (including negatives)
FLOAT_PATTERN = re.compile(r"-?\d+\.\d+")


def remove_comments(svg: str) -> str:
    return COMMENT_PATTERN.sub("", svg)


def remove_metadata(svg: str) -> str:
    try:
        root = etree.fromstring(svg.encode("utf-8"))
        for elem in root.xpath("//metadata"):
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
        return etree.tostring(root, encoding="unicode")
    except:
        return svg


def normalize_floats(svg: str, precision: int = 1) -> str:
    def repl(match):
        return str(round(float(match.group()), precision))
    return FLOAT_PATTERN.sub(repl, svg)


def is_valid(svg: str) -> bool:
    try:
        etree.fromstring(svg.encode("utf-8"))
        return True
    except:
        return False


def is_reasonable_length(svg: str, min_len=50, max_len=2048) -> bool:
    return min_len <= len(svg) <= max_len


def canonicalize(svg: str) -> str:
    try:
        root = etree.fromstring(svg.encode("utf-8"))
        return etree.tostring(root, encoding="unicode")
    except:
        return svg


def clean_svg(svg: str) -> str:
    svg = remove_comments(svg)
    svg = remove_metadata(svg)
    svg = normalize_floats(svg)
    svg = svg.strip()
    return svg
