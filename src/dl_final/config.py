"""Small config loader for the repository's simple YAML files.

The Sprint 1 runtime intentionally avoids a hard dependency on PyYAML so dataset
audit scripts can run in lightweight environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load `configs/dataset/selected_dataset.yaml`.

    This parser supports the subset used by the repo config: nested dictionaries,
    scalar values, and lists of scalar values.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_list_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "#" in raw_line:
            raw_line = raw_line.split("#", 1)[0].rstrip()

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if line.startswith("- "):
            if pending_list_key is None:
                raise ValueError(f"Unexpected list item in {config_path}: {line}")
            list_indent, parent, key = pending_list_key
            if indent <= list_indent:
                raise ValueError(f"Invalid list indentation in {config_path}: {line}")
            if parent.get(key) == {}:
                parent[key] = []
            if not isinstance(parent.get(key), list):
                raise ValueError(f"Config key is not a list: {key}")
            parent[key].append(_parse_scalar(line[2:].strip()))
            continue

        pending_list_key = None
        if ":" not in line:
            raise ValueError(f"Invalid config line in {config_path}: {line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation in {config_path}: {line}")

        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"Cannot assign key under non-dict parent: {line}")

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            pending_list_key = (indent, parent, key)
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
