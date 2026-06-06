"""Safe execution of LLM-generated pandas code.

Sandboxing (file I/O, network, subprocess isolation) is deferred to Week 3.
This module handles *reliability* guards: timeout and row-count limits.
"""

import threading
from typing import Any, Optional

import pandas as pd

# Builtins available inside generated code — no __import__, no open, no eval.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "None": None, "print": print, "range": range,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "True": True, "False": False, "tuple": tuple, "type": type, "zip": zip,
}


def execute_code(
    code: str,
    df: pd.DataFrame,
    timeout: int = 30,
    max_rows: int = 50_000,
) -> tuple[Optional[Any], Optional[str]]:
    """Execute generated code in a restricted environment.

    Returns ``(result, error_message)``. On success ``error_message`` is None;
    on failure ``result`` is None.

    The thread runs as a daemon so it cannot block process exit if it outlives
    the timeout (relevant for truly infinite loops in tests or edge cases).
    """
    if len(df) > max_rows:
        df = df.head(max_rows)

    result_holder: dict[str, Any] = {}
    error_holder: dict[str, str] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            globs: dict[str, Any] = {
                "__builtins__": _SAFE_BUILTINS,
                "pd": pd,
                "df": df,
            }
            locs: dict[str, Any] = {}
            exec(compile(code, "<generated>", "exec"), globs, locs)  # noqa: S102
            result_holder["value"] = locs.get("result")
        except Exception as exc:
            error_holder["msg"] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    if not done.wait(timeout=timeout):
        return None, f"TimeoutError: execution exceeded {timeout}s"

    if "msg" in error_holder:
        return None, error_holder["msg"]

    return result_holder.get("value"), None
