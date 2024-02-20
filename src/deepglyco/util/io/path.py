__all__ = ["list_files"]

import os
import re
import itertools


def list_files(path=".", pattern=None, recursive=False, include_dirs=False):
    if recursive:
        return list(
            itertools.chain.from_iterable(
                [
                    os.path.join(t[0], x)
                    for x in (t[1] + t[2] if include_dirs else t[2])
                    if pattern is None or re.search(pattern, x) is not None
                ]
                for t in os.walk(path)
            )
        )
    else:
        return [
            t
            for t in os.listdir(path)
            if pattern is None or re.search(pattern, t) is not None
        ]
