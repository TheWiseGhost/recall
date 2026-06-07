"""``${VAR}`` interpolation for YAML configuration.

Supported forms:

``${VAR}``            required; raises if unset
``${VAR:-default}``   falls back to ``default`` when unset or empty
``${VAR-default}``    falls back to ``default`` only when unset

Secrets therefore live in the environment and configuration files stay safe to
commit.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from recall.core.errors import ConfigurationError

_PATTERN = re.compile(
    r"""\$\{
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)
        (?:
            (?P<op>:-|-)
            (?P<default>[^}]*)
        )?
    \}""",
    re.VERBOSE,
)


def interpolate(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` references in strings, lists and dicts."""
    environ = os.environ if env is None else env

    if isinstance(value, str):
        return _expand_string(value, environ)
    if isinstance(value, dict):
        return {key: interpolate(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate(item, environ) for item in value]
    return value


def _expand_string(text: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        op = match.group("op")
        default = match.group("default")
        present = name in env
        current = env.get(name, "")

        if op == ":-":
            return current if current else (default or "")
        if op == "-":
            return current if present else (default or "")
        if not present:
            raise ConfigurationError(
                f"Environment variable {name!r} is referenced by the configuration "
                f"but is not set. Set it, or give it a default: ${{{name}:-...}}"
            )
        return current

    return _PATTERN.sub(replace, text)
