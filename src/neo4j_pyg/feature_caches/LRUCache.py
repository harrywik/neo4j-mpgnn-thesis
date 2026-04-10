from collections import OrderedDict

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class LRUCache(Neo4jCache):
    """Fixed-capacity LRU-eviction cache."""

    def __init__(self, max_entries: int) -> None:
        self._data: OrderedDict = OrderedDict()
        self._max = max_entries

    def get(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def set(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._max:
            self._data.popitem(last=False)

    def delete(self, key):
        self._data.pop(key, None)

    def clear(self):
        self._data.clear()
