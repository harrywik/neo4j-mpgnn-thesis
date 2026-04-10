from typing import Dict, Optional

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class StaticCache(Neo4jCache):
    """Write-once cache.  After :meth:`freeze` is called, writes are ignored."""

    def __init__(self, data: Optional[Dict] = None) -> None:
        self._data: Dict = dict(data) if data else {}
        self._frozen = bool(data)

    def freeze(self) -> None:
        self._frozen = True

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        if not self._frozen:
            self._data[key] = value

    def delete(self, key):
        if not self._frozen:
            self._data.pop(key, None)

    def clear(self):
        if not self._frozen:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
