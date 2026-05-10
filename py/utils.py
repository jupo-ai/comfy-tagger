from pathlib import Path


AUTHOR = "jupo"
ROOT_DIR = Path(__file__).parent.parent


def mk_name(*args):
    parts = [AUTHOR] + list(args)
    return ".".join(parts)


def mk_category(*args):
    parts = [AUTHOR] + list(args)
    return "/".join(parts)
