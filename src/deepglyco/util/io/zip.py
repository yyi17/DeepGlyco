__all__ = ["zip_content"]

import os
from typing import Literal, Optional, Tuple

try:
    from zipfile_deflate64 import ZipFile
except ModuleNotFoundError:
    from zipfile import ZipFile


def resolve_zip_content_path(path: str) -> Tuple[str, Optional[str]]:
    if os.path.exists(path):
        return path, None

    parent_path = path
    content_path = None

    while (
        not os.path.exists(parent_path)
        or os.path.splitext(parent_path)[1].lower() != ".zip"
    ):
        parent, basename = os.path.split(parent_path)
        if parent == "" or basename == "":
            raise IOError(f"file not found: {path}")

        parent_path = parent
        if content_path is None:
            content_path = basename
        else:
            content_path = f"{basename}/{content_path}"

    return parent_path, content_path


def zip_content(path: str, mode: Literal["r"] = "r"):
    parent_path, content_path = resolve_zip_content_path(path)
    if content_path is None:
        return open(parent_path, mode)

    zip = ZipFile(parent_path)
    return zip.open(content_path, mode)
