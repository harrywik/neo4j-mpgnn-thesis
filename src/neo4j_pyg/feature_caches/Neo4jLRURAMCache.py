"""In-process LRU cache backed by an OrderedDict.

Pure in-memory, zero-dependency LRU.  Intended as the dynamic write-back
tier in a :class:`TieredCache` — DB-fetched entries are written here
automatically via TieredCache.set_many so repeated accesses are served
from RAM.
"""

from collections import OrderedDict
from typing import Dict

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class Neo4jLRURAMCache(Neo4jCache):
    """Fixed-capacity in-memory LRU cache.

    Parameters
    ----------
    max_entries:
        Maximum number of entries before LRU eviction kicks in.
    """

    def __init__(self, max_entries: int = 100_000) -> None:
        self._data: OrderedDict = OrderedDict()
        self._max = max_entries

    def get(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def get_many(self, keys) -> Dict:
        result = {}
        for k in keys:
            v = self._data.get(k)
            if v is not None:
                self._data.move_to_end(k)
                result[k] = v
        return result

    def set(self, key, value) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._max:
            self._data.popitem(last=False)

    def set_many(self, items: dict) -> None:
        for k, v in items.items():
            self.set(k, v)

    def delete(self, key) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
