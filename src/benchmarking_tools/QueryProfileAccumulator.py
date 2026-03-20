from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Stateless extraction helpers
# ---------------------------------------------------------------------------

def flatten_profile(node: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    """Recursively flatten a Neo4j PROFILE plan tree into a flat list.

    Each entry corresponds to one plan operator.  ``time_ns`` is the raw
    nanosecond value stored by Neo4j (divide by 1 000 000 to get ms).
    """
    args = node.get("args", {})
    out.append({
        "id": args.get("Id"),
        "operator": node.get("operatorType"),
        "details": args.get("Details"),
        "rows": node.get("rows") if node.get("rows") is not None else args.get("Rows"),
        "db_hits": node.get("dbHits") if node.get("dbHits") is not None else args.get("DbHits"),
        "time_ns": node.get("time") if node.get("time") is not None else args.get("Time"),
        "memory_bytes": args.get("Memory"),
        "page_cache_hits": (
            node.get("pageCacheHits")
            if node.get("pageCacheHits") is not None
            else args.get("PageCacheHits")
        ),
        "page_cache_misses": (
            node.get("pageCacheMisses")
            if node.get("pageCacheMisses") is not None
            else args.get("PageCacheMisses")
        ),
        "estimated_rows": args.get("EstimatedRows"),
        "pipeline": args.get("PipelineInfo"),
    })
    for child in node.get("children", []):
        flatten_profile(child, out)


def extract_query_metrics(
    summary: Any,
    t_send_query: float,
    t_all_records: float,
) -> Dict[str, Any]:
    """Combine Neo4j driver summary fields with client-side wall-clock timings.

    Three distinct time metrics are captured — they measure different things:

    * ``client_wall_time_ms``:   full round-trip including network + Python
      record materialisation.
    * ``result_consumed_after_ms``:  DB-side end-to-end query time reported
      by the driver — the cleanest single DB-latency metric.
    * ``result_available_after_ms``:  time until the first result record was
      ready on the server.
    * ``profile_total_time_ms``:  root operator ``args["Time"]`` converted
      from nanoseconds; represents the top-level operator cost only.
    """
    profile = summary.profile or {}
    args = profile.get("args", {}) if profile else {}

    operators: List[Dict[str, Any]] = []
    if profile:
        flatten_profile(profile, operators)

    return {
        "client": {
            "client_wall_time_ms": (t_all_records - t_send_query) * 1000,
        },
        "driver": {
            "result_available_after_ms": summary.result_available_after,
            "result_consumed_after_ms": summary.result_consumed_after,
        },
        "plan": {
            "operator_type": profile.get("operatorType") if profile else None,
            "rows": profile.get("rows") if profile else None,
            "db_hits": profile.get("dbHits") if profile else None,
            # args["Time"] is in nanoseconds; divide by 1 000 000 → ms
            "profile_total_time_ms": (args.get("Time") or 0) / 1_000_000,
            "page_cache_hits": (
                profile.get("pageCacheHits") if profile else args.get("PageCacheHits")
            ),
            "page_cache_misses": (
                profile.get("pageCacheMisses") if profile else args.get("PageCacheMisses")
            ),
            "global_memory_bytes": args.get("GlobalMemory"),
            "runtime": args.get("runtime"),
            "runtime_impl": args.get("runtime-impl"),
            "runtime_version": args.get("runtime-version"),
            "planner": args.get("planner"),
            "planner_impl": args.get("planner-impl"),
            "planner_version": args.get("planner-version"),
            "batch_size": args.get("batch-size"),
        },
        "operators": operators,
    }


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class QueryProfileAccumulator:
    """Accumulates Neo4j PROFILE query metrics across many samples.

    Each call to :meth:`add` receives the Neo4j driver ``ResultSummary``
    object plus client-side wall-clock timestamps.  Running sums are kept in
    memory (O(operators), never O(samples)).  Call :meth:`get_summary` or
    :meth:`save` at the end of training to obtain averaged results.

    Usage::

        acc = QueryProfileAccumulator()
        # inside training loop, after each DB call:
        acc.add(summary, "sampler", t_send, t_all_records)
        acc.add(summary, "feat_x",  t_send, t_all_records)
        acc.add(summary, "feat_y",  t_send, t_all_records)
        # at the end:
        acc.save(run_dir / "query_profile.json", subphase_metrics=summary_dict)
    """

    def __init__(self) -> None:
        # source → { count, global_sums, operators, plan_meta }
        self._data: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        summary: Any,
        source: str,
        t_send_query: float,
        t_all_records: float,
    ) -> None:
        """Incorporate one query execution into the running sums.

        ``summary`` is the object returned by ``result.consume()`` from the
        Neo4j Python driver.  If profiling was not active ``summary.profile``
        will be ``None``; global driver timings are still accumulated.
        """
        metrics = extract_query_metrics(summary, t_send_query, t_all_records)

        if source not in self._data:
            self._data[source] = {
                "count": 0,
                "global_sums": {
                    "client_wall_time_ms": 0.0,
                    "result_available_after_ms": 0.0,
                    "result_consumed_after_ms": 0.0,
                    "profile_total_time_ms": 0.0,
                    "global_memory_bytes": 0.0,
                    "root_rows": 0.0,
                    "root_db_hits": 0.0,
                    "page_cache_hits": 0.0,
                    "page_cache_misses": 0.0,
                },
                # operator_id → running sums dict
                "operators": {},
                # store latest plan metadata (planner/runtime don't change)
                "plan_meta": {},
            }

        src = self._data[source]
        src["count"] += 1
        glb = src["global_sums"]

        c = metrics["client"]
        d = metrics["driver"]
        p = metrics["plan"]

        glb["client_wall_time_ms"] += c["client_wall_time_ms"] or 0.0
        glb["result_available_after_ms"] += d["result_available_after_ms"] or 0.0
        glb["result_consumed_after_ms"] += d["result_consumed_after_ms"] or 0.0
        glb["profile_total_time_ms"] += p["profile_total_time_ms"] or 0.0
        glb["global_memory_bytes"] += p["global_memory_bytes"] or 0.0
        glb["root_rows"] += p["rows"] or 0.0
        glb["root_db_hits"] += p["db_hits"] or 0.0
        glb["page_cache_hits"] += p["page_cache_hits"] or 0.0
        glb["page_cache_misses"] += p["page_cache_misses"] or 0.0

        # Store plan metadata from the first sample that has it
        if not src["plan_meta"] and any(p.get(k) for k in ("runtime", "planner")):
            src["plan_meta"] = {
                "runtime": p.get("runtime"),
                "runtime_impl": p.get("runtime_impl"),
                "runtime_version": p.get("runtime_version"),
                "planner": p.get("planner"),
                "planner_impl": p.get("planner_impl"),
                "planner_version": p.get("planner_version"),
                "batch_size": p.get("batch_size"),
            }

        self._accumulate_operators(metrics["operators"], src["operators"])

    def has_data(self) -> bool:
        return bool(self._data)

    def get_summary(self) -> Dict[str, Any]:
        """Return averaged metrics for all sources collected so far."""
        result: Dict[str, Any] = {}

        for source, data in self._data.items():
            count = data["count"]
            if count == 0:
                continue

            glb = data["global_sums"]
            avg_global = {
                "sample_count": count,
                "avg_client_wall_time_ms": glb["client_wall_time_ms"] / count,
                "avg_result_available_after_ms": glb["result_available_after_ms"] / count,
                "avg_result_consumed_after_ms": glb["result_consumed_after_ms"] / count,
                "avg_profile_total_time_ms": glb["profile_total_time_ms"] / count,
                "avg_global_memory_bytes": glb["global_memory_bytes"] / count,
                "avg_root_rows": glb["root_rows"] / count,
                "avg_root_db_hits": glb["root_db_hits"] / count,
                "avg_page_cache_hits": glb["page_cache_hits"] / count,
                "avg_page_cache_misses": glb["page_cache_misses"] / count,
            }

            avg_operators = []
            for op_id, op in sorted(data["operators"].items()):
                n = op["count"]
                if n == 0:
                    continue
                avg_operators.append({
                    "id": op_id,
                    "operator_type": op["operator_type"],
                    "details": op["details"],
                    "pipeline": op["pipeline"],
                    "avg_time_ms": (op["time_ns_sum"] / n) / 1_000_000,
                    "avg_db_hits": op["db_hits_sum"] / n,
                    "avg_rows": op["rows_sum"] / n,
                    "avg_estimated_rows": op["estimated_rows_sum"] / n,
                    "avg_memory_bytes": op["memory_bytes_sum"] / n,
                    "avg_page_cache_hits": op["page_cache_hits_sum"] / n,
                    "avg_page_cache_misses": op["page_cache_misses_sum"] / n,
                })

            # Sum of all individual operator avg times (kept for operator-level plots).
            avg_operators_time_sum_ms = sum(op["avg_time_ms"] for op in avg_operators)
            avg_global["avg_db_exec_time_ms"] = avg_operators_time_sum_ms

            # Legacy derived metrics (kept for backward compatibility).
            avg_global["avg_network_transfer_ms"] = (
                avg_global["avg_result_consumed_after_ms"] - avg_operators_time_sum_ms
            )
            avg_global["avg_driver_overhead_ms"] = (
                avg_global["avg_client_wall_time_ms"] - avg_operators_time_sum_ms
            )

            # Clean 4-segment breakdown based on Bolt protocol timestamps:
            #   t_first (result_available_after): server hands off first byte to TCP.
            #   t_last  (result_consumed_after):  server hands off last byte to TCP.
            #   wall    (client_wall_time):        Python finishes receiving all records.
            t_first = avg_global.get("avg_result_available_after_ms") or 0.0
            t_last  = avg_global.get("avg_result_consumed_after_ms")  or 0.0
            wall    = avg_global.get("avg_client_wall_time_ms")        or 0.0
            # Time from query send to server starting to stream: parse + plan + pipeline init.
            avg_global["avg_query_startup_ms"]   = max(0.0, t_first)
            # Time from first byte to last byte leaving the server: operator execution + Bolt serialization.
            avg_global["avg_exec_serialize_ms"]  = max(0.0, t_last - t_first)
            # Time from server done to Python done: network transit + Python Bolt deserialization.
            avg_global["avg_client_recv_ms"]     = max(0.0, wall - t_last)

            result[source] = {
                "plan_meta": data["plan_meta"],
                "global": avg_global,
                "operators": avg_operators,
                "top_by_time_ms": sorted(
                    avg_operators, key=lambda x: x["avg_time_ms"], reverse=True
                )[:5],
                "top_by_db_hits": sorted(
                    avg_operators, key=lambda x: x["avg_db_hits"], reverse=True
                )[:5],
                "top_by_memory_bytes": sorted(
                    avg_operators, key=lambda x: x["avg_memory_bytes"], reverse=True
                )[:5],
            }

        return result

    def save(
        self,
        path: Path,
        subphase_metrics: Optional[dict] = None,
    ) -> None:
        """Write averaged profiling results to *path* as JSON.

        If *subphase_metrics* is provided (the ``"metrics"`` section from
        ``measurements.json``), it is embedded under the ``"subphase_metrics"``
        key so both sources of timing data live in one file.
        """
        summary = self.get_summary()
        if subphase_metrics is not None:
            summary["subphase_metrics"] = subphase_metrics
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accumulate_operators(
        self, operators: List[Dict[str, Any]], ops: dict
    ) -> None:
        """Add one query's flat operator list into the running-sum dict."""
        for op in operators:
            op_id = op.get("id")
            if op_id is None:
                continue
            if op_id not in ops:
                ops[op_id] = {
                    "count": 0,
                    "time_ns_sum": 0,
                    "db_hits_sum": 0,
                    "rows_sum": 0,
                    "estimated_rows_sum": 0.0,
                    "memory_bytes_sum": 0,
                    "page_cache_hits_sum": 0,
                    "page_cache_misses_sum": 0,
                    "operator_type": op.get("operator") or "",
                    "details": op.get("details") or "",
                    "pipeline": op.get("pipeline") or "",
                }
            entry = ops[op_id]
            entry["count"] += 1
            entry["time_ns_sum"] += op.get("time_ns") or 0
            entry["db_hits_sum"] += op.get("db_hits") or 0
            entry["rows_sum"] += op.get("rows") or 0
            entry["estimated_rows_sum"] += op.get("estimated_rows") or 0.0
            entry["memory_bytes_sum"] += op.get("memory_bytes") or 0
            entry["page_cache_hits_sum"] += op.get("page_cache_hits") or 0
            entry["page_cache_misses_sum"] += op.get("page_cache_misses") or 0
