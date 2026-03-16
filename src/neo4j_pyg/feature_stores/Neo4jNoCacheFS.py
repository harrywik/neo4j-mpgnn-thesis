from pathlib import Path
import sys
from typing import Optional

from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS


class Neo4jNoCacheFS(Neo4jAbstractFS):
    """No-cache Neo4j feature store that implements :class:`Neo4jAbstractFS`.

    Every feature request goes directly to the database — no data is retained
    between calls. All fetch logic (sorting, timing, pre-allocated arrays) is
    inherited from :meth:`Neo4jAbstractFS._get_value_from_db`.
    """

    def _get_cached_value(
        self, nid: int, attr: TensorAttr, **kwargs
    ) -> Optional[object]:
        return None

    def _update_cached_value(
        self, nid: int, value: object, attr: TensorAttr, **kwargs
    ) -> None:
        pass

    def _remove_cached_value(
        self, nid: int, attr: TensorAttr, **kwargs
    ) -> None:
        pass
