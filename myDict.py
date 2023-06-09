from collections.abc import Hashable
from typing import Tuple


class Dict2D:
    def __init__(self, keys: Tuple[Tuple[Hashable, ...], ...], values: Tuple[...]):
        if len(keys) != len(values):
            raise Exception('Len of key sets not match len of values!')

    def


