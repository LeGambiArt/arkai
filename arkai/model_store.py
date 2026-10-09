"""Structured inventory for native runtime model installations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    """An installed selection and the source files it depends on."""

    reference: str
    installation_id: str
    path: Path
    complete: bool | None
    sources: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    revision: str | None = None
    owned_bytes: int = 0
    shared_bytes: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ModelEntry":
        """Validate inventory received from a versioned installer bridge."""
        if (
            not isinstance(value, dict)
            or not all(
                isinstance(value.get(key), str) and value[key]
                for key in ("reference", "installation_id", "path")
            )
            or not isinstance(value.get("complete"), bool)
            or not isinstance(value.get("sources", {}), dict)
            or not all(isinstance(source, dict) for source in value.get("sources", {}).values())
            or not isinstance(value.get("files", []), list)
            or not all(isinstance(path, str) for path in value.get("files", []))
            or any(
                type(value.get(key, 0)) is not int or value.get(key, 0) < 0
                for key in ("owned_bytes", "shared_bytes")
            )
            or value.get("revision") is not None
            and not isinstance(value["revision"], str)
        ):
            raise RuntimeError("Splash bridge returned an invalid model inventory")
        return cls(
            reference=value["reference"],
            installation_id=value["installation_id"],
            path=Path(value["path"]),
            complete=value["complete"],
            sources=value.get("sources", {}),
            files=value.get("files", []),
            revision=value.get("revision"),
            owned_bytes=value.get("owned_bytes", 0),
            shared_bytes=value.get("shared_bytes", 0),
        )
