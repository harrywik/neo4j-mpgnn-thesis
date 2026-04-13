from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional


class Neo4jCache(ABC):
    """Minimal key-value cache contract.

    Subclasses implement get/set/delete/clear.  Complex caching strategies
    are built by *composing* simple Neo4jCache instances (see :class:`TieredCache`)
    rather than deep inheritance hierarchies.

    Key format is ``(attr_name, node_id)`` where ``attr_name`` is ``"x"``
    (features) or ``"y"`` (labels).

    ``static = True`` signals to :class:`TieredCache` that this tier should
    not be written to during write-through or promotion.
    """

    static: bool = False

    @abstractmethod
    def get(self, key: Any) -> Optional[Any]:
        """Return value for *key*, or ``None`` on miss."""

    @abstractmethod
    def set(self, key: Any, value: Any) -> None:
        ...

    @abstractmethod
    def delete(self, key: Any) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    def get_many(self, keys: Iterable[Any]) -> Dict[Any, Any]:
        """Return ``{key: value}`` for all cached keys.

        Override for batch-aware backends (Redis MGET, etc.).
        """
        return {k: v for k in keys if (v := self.get(k)) is not None}

    def set_many(self, items: Dict[Any, Any]) -> None:
        """Write all key-value pairs.

        Override for batch-aware backends (Redis pipeline, etc.).
        """
        for k, v in items.items():
            self.set(k, v)

    def fill_from_neo4j(self, uri: str, user: str, pwd: str, **kwargs) -> None:
        """Pre-fill the cache from Neo4j.

        Default is a no-op — caches that support eager prefill override this.
        :class:`TieredCache` propagates the call to every tier.
        """

    def __contains__(self, key: Any) -> bool:
        return self.get(key) is not None
