"""Curses field editor for configuration files.

Data-driven scalar-field registry plus pure, UI-free helpers so all logic is
unit-testable; the curses UI is isolated in ``run_editor``. Scalar registry
keys and the ``inference.profiles`` / ``sandbox.profiles`` collections are
edited through ``CollectionSpec`` modals; all mutations land in the loaded
target dict, which the caller writes back to disk.
"""

import copy
import dataclasses
from typing import Any

import yaml

from arkai import utils
from arkai.config import DEFAULTS
from arkai.sandbox import _is_valid_profile_name

UNSET_LABEL = "(unset)"


class _Missing:
    """Sentinel for absent nested keys."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<MISSING>"


_MISSING = _Missing()


@dataclasses.dataclass(frozen=True)
class Field:
    """One editable scalar configuration field."""

    key: str
    label: str
    kind: str  # "choice" | "bool" | "int" | "float" | "str" | "context"
    choices: tuple[str, ...] | None = None
    dynamic: str | None = None
    minimum: float | None = None
    maximum: float | None = None


def _sampling_field(name: str) -> Field:
    return Field(f"inference.{name}", name.replace("_", " ").capitalize(), "float", minimum=0)


FIELD_GROUPS: dict[str, list[Field]] = {
    "Agent": [
        Field("agent.name", "Name", "choice", dynamic="agents"),
        Field("agent.mode", "Mode", "choice", choices=("local", "cloud")),
        Field("agent.path", "Path", "str"),
        Field("agent.mcp", "Mcp", "bool"),
        Field("agent.theme", "Theme", "str"),
        Field("agent.cloud.default", "Cloud", "choice", choices=("openai", "vertex", "custom")),
        Field("agent.cloud.custom.default_provider", "Default provider", "str"),
        Field("agent.cloud.custom.protocol", "Custom protocol", "protocol"),
        Field("agent.cloud.custom.base_url", "Custom base url", "url"),
        Field("agent.cloud.custom.api_key", "Custom api key", "api_key"),
    ],
    "Inference": [
        Field("inference.backend", "Backend", "choice", dynamic="backends"),
        Field("inference.model", "Model", "str"),
        Field("inference.path", "Path", "str"),
        Field("inference.port", "Port", "int", minimum=1024, maximum=65535),
        Field("inference.gpu_layers", "Gpu layers", "int", minimum=-1),
        Field("inference.context_size", "Context size", "context"),
        Field("inference.startup_timeout", "Startup timeout", "int", minimum=1),
        _sampling_field("temperature"),
        _sampling_field("top_p"),
        _sampling_field("top_k"),
        _sampling_field("min_p"),
        _sampling_field("presence_penalty"),
        _sampling_field("frequency_penalty"),
        _sampling_field("repetition_penalty"),
    ],
    "WTMCP": [
        Field("wtmcp.enabled", "Enabled", "bool"),
        Field("wtmcp.path", "Path", "str"),
        Field("wtmcp.port", "Port", "int", minimum=1024, maximum=65535),
        Field("wtmcp.workdir", "Workdir", "str"),
        Field("wtmcp.config", "Config", "str"),
    ],
    "Sandbox": [
        Field("sandbox.enabled", "Enabled", "bool"),
        Field("sandbox.disable", "Disable", "bool"),
        Field("sandbox.path", "Path", "str"),
        Field("sandbox.memory_mb", "Memory MB", "int", minimum=1),
        Field("sandbox.cpus", "Cpus", "int", minimum=1),
        Field("sandbox.pids", "Pids", "int", minimum=1),
        Field("sandbox.timeout", "Timeout", "int", minimum=0),
        Field("sandbox.active_profile", "Active profile", "str"),
    ],
    "Vector database": [
        Field("vectordb.vendor", "Vendor", "choice", choices=("qdrant", "chromadb")),
        Field("vectordb.engine", "Engine", "choice", choices=("auto", "podman", "apple", "docker")),
        Field("vectordb.port", "Port", "int", minimum=1024, maximum=65535),
        Field("vectordb.image", "Image", "str"),
        Field("vectordb.database_dir", "Database dir", "str"),
    ],
    "RAG": [
        Field("rag.chunk_size", "Chunk size", "int", minimum=1),
        Field("rag.chunk_overlap", "Chunk overlap", "int", minimum=0),
        Field("rag.embedding_model", "Embedding model", "str"),
        Field("rag.max_search_results", "Max search results", "int", minimum=1),
    ],
}

ALL_FIELDS: tuple[Field, ...] = tuple(field for fields in FIELD_GROUPS.values() for field in fields)


@dataclasses.dataclass(frozen=True)
class CollectionSpec:
    """One profile collection edited through a dedicated modal."""

    title: str  # "Model profiles" | "Sandbox profiles"
    prefix: str  # "inference.profiles" | "sandbox.profiles"
    members: tuple[Field, ...]  # member-local Field keys ("model", "memory_mb", ...)
    required: tuple[str, ...]  # member keys that must be non-empty on creation


def _member(name: str, kind: str, minimum: float | None = None) -> Field:
    return Field(name, name.replace("_", " ").capitalize(), kind, minimum=minimum)


INFERENCE_PROFILE_MEMBERS: tuple[Field, ...] = tuple(
    _member(name, "float", minimum=0) if name != "model" else _member(name, "str")
    for name in (
        "model",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    )
)

SANDBOX_PROFILE_MEMBERS: tuple[Field, ...] = (
    _member("path", "str"),
    _member("memory_mb", "int", minimum=1),
    _member("cpus", "int", minimum=1),
    _member("pids", "int", minimum=1),
    _member("timeout", "int", minimum=0),
)

CLOUD_PROVIDER_MEMBERS: tuple[Field, ...] = (
    Field("name", "Name", "str"),
    Field("protocol", "Protocol", "protocol"),
    Field("base_url", "Base url", "url"),
    Field("default_model", "Default model", "str"),
)

COLLECTIONS: list[CollectionSpec] = [
    CollectionSpec(
        title="Model profiles",
        prefix="inference.profiles",
        members=INFERENCE_PROFILE_MEMBERS,
        required=("model",),
    ),
    CollectionSpec(
        title="Sandbox profiles",
        prefix="sandbox.profiles",
        members=SANDBOX_PROFILE_MEMBERS,
        required=(),
    ),
    CollectionSpec(
        title="Cloud custom providers",
        prefix="agent.cloud.custom.providers",
        members=CLOUD_PROVIDER_MEMBERS,
        required=(),
    ),
]


def _dynamic_choices(name: str) -> tuple[str, ...]:
    """Resolve a dynamic choice provider by name (lazy imports avoid cycles)."""
    if name == "agents":
        from arkai import config

        return tuple(sorted(config.VALID_AGENTS))
    if name == "backends":
        from arkai import inference_backend

        return tuple(sorted(inference_backend.get_backend_names()))
    raise ValueError(f"Unknown dynamic choice provider: {name}")


def field_choices(field: Field) -> tuple[str, ...]:
    """Return the resolved choice list for a choice field."""
    if field.choices is not None:
        return field.choices
    return _dynamic_choices(field.dynamic or "")


# --- Pure helpers ---------------------------------------------------------


def nested_get(d: dict, key: str) -> Any:
    """Return the value at a dotted key, or ``_MISSING`` when absent."""
    current: Any = d
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def nested_set(d: dict, key: str, value: Any) -> None:
    """Set the value at a dotted key, creating intermediate dicts."""
    parts = key.split(".")
    current = d
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def nested_delete(d: dict, key: str) -> None:
    """Remove the value at a dotted key; prune now-empty parent dicts."""
    parts = key.split(".")
    chain: list[dict] = [d]
    for part in parts[:-1]:
        child = chain[-1].get(part)
        if not isinstance(child, dict) or part not in chain[-1]:
            return
        chain.append(child)
    if chain[-1].pop(parts[-1], None) is None:
        return  # nothing deleted; leave empty parents untouched
    for parent, part in zip(reversed(chain[:-1]), reversed(parts[:-1])):
        child = parent.get(part)
        if isinstance(child, dict) and not child:
            del parent[part]
        else:
            break


def key_present(d: dict, key: str) -> bool:
    """Return whether the dotted key exists in the dict."""
    return nested_get(d, key) is not _MISSING


def _layer_source(user: dict, project: dict, key: str) -> str:
    """Return the effective-value source layer for one dotted key."""
    if key_present(project, key):
        return "project"
    if key_present(user, key):
        return "global"
    return "default"


def resolve_field_sources(user: dict, project: dict) -> dict[str, str]:
    """Map every registry key to its effective-value source layer."""
    return {field.key: _layer_source(user, project, field.key) for field in ALL_FIELDS}


def validate_provider_id(existing: list[str], provider_id: str) -> str | None:
    """Validate a proposed custom provider ID; None when valid, else an error."""
    from arkai.cloud_provider import _PROVIDER_ID_PATTERN

    if not provider_id:
        return "ID must not be empty"
    if not _PROVIDER_ID_PATTERN.fullmatch(provider_id):
        return "ID must contain only letters, digits, '_' or '-'"
    if provider_id in existing:
        return f"provider {provider_id!r} already exists"
    return None


def validate_provider_field(field: Field, raw: str) -> tuple[bool, str, Any]:
    """Validate provider-member input that needs cloud-specific rules."""
    if field.kind == "protocol":
        from arkai.cloud_provider import _SUPPORTED_CUSTOM_PROTOCOLS

        if raw in _SUPPORTED_CUSTOM_PROTOCOLS:
            return True, "", raw
        return False, f"must be one of {', '.join(sorted(_SUPPORTED_CUSTOM_PROTOCOLS))}", None
    if field.kind == "url":
        from arkai.cloud_provider import _normalize_base_url

        if raw == "":
            return True, "", ""
        try:
            url, _host = _normalize_base_url(raw, field.key)
        except ValueError as error:
            return False, str(error).replace(f"{field.key} ", ""), None
        return True, "", url
    return validate_input(field, raw)


def validate_custom_key(key: str) -> str | None:
    """Validate a dotted custom key name; None when valid, else an error."""
    if not key:
        return "key must not be empty"
    if any(not part for part in key.split(".")):
        return "key segments must not be empty"
    if key in {f.key for f in ALL_FIELDS} or key in {spec.prefix for spec in COLLECTIONS}:
        return "key is already managed by the editor; edit it in its group instead"
    return None


def parse_custom_value(raw: str) -> tuple[bool, str, Any]:
    """Parse a custom value as YAML; returns (ok, error, parsed value)."""
    if raw == "":
        return False, "value must not be empty", None
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        return False, f"invalid YAML: {str(error).splitlines()[0]}", None
    return True, "", value


def validate_profile_name(existing: list[str], name: str) -> str | None:
    """Validate a proposed profile name; None when valid, else an error."""
    if not _is_valid_profile_name(name):
        return (
            "name must be alphanumeric/underscore and not 'default' or 'active'"
            if name
            else "name must not be empty"
        )
    if name in existing:
        return f"profile {name!r} already exists"
    return None


def validate_volume_entry(existing: list, entry: str) -> str | None:
    """Validate one volume entry against the existing list; None when valid."""
    return utils.validate_volumes(existing + [entry])


def validate_env_entry(raw: str) -> tuple[bool, str, str, str]:
    """Parse and validate a ``KEY=VALUE`` entry.

    Returns ``(ok, message, key, value)``; key and value are empty when
    parsing failed.
    """
    key, separator, value = raw.partition("=")
    if not separator:
        return False, "entry must be in KEY=VALUE form", "", ""
    if not key:
        return False, "key must not be empty", "", ""
    message = utils.validate_environment({key: value})
    if message:
        return False, message, "", ""
    return True, "", key, value


def validate_input(field: Field, raw: str) -> tuple[bool, str, Any]:
    """Validate raw input text against a field.

    Returns ``(ok, error_message, parsed_value)``.
    """
    if field.kind == "choice":
        choices = field_choices(field)
        if raw in choices:
            return True, "", raw
        return False, f"must be one of {', '.join(choices)}", None
    if field.kind == "bool":
        if raw == "true":
            return True, "", True
        if raw == "false":
            return True, "", False
        return False, "must be true or false", None
    if field.kind == "int":
        try:
            value = int(raw)
        except ValueError:
            return False, f"must be an integer, got {raw!r}", None
        return _check_bounds(field, value, raw)
    if field.kind == "float":
        try:
            value = float(raw)
        except ValueError:
            return False, f"must be a number, got {raw!r}", None
        return _check_bounds(field, value, raw)
    if field.kind == "context":
        parsed = utils.parse_token_size(raw)
        if parsed is None:
            return False, f"must be a positive integer or k/m size, got {raw!r}", None
        # Store the raw string: the loader/validator already accept suffixes.
        return True, "", raw
    return True, "", raw  # str


def _check_bounds(field: Field, value: float, raw: str) -> tuple[bool, str, Any]:
    if field.minimum is not None and value < field.minimum:
        return False, f"must be >= {field.minimum}, got {raw}", None
    if field.maximum is not None and value > field.maximum:
        return False, f"must be <= {field.maximum}, got {raw}", None
    return True, "", value


def allows_unset(field: Field) -> bool:
    """Whether a choice/bool field whose default is None offers "(unset)"."""
    return field.kind in ("choice", "bool") and nested_get(DEFAULTS, field.key) is None


def effective_default(field: Field) -> Any:
    """Return the built-in default for a field (None when the key is absent)."""
    value = nested_get(DEFAULTS, field.key)
    return None if value is _MISSING else value


# --- Curses UI ------------------------------------------------------------

FOOTER = "↑/↓ move · Enter edit · a add key · d delete key · r reset · s save · q quit · ? help"


def _interactive_terminal() -> bool:
    import sys

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    import os

    return bool(os.environ.get("TERM"))


def run_editor(
    target: dict,
    merged: dict,
    user: dict,
    project: dict,
    target_path: str,
) -> bool:
    """Run the curses field editor.

    Mutates ``target`` in place on every accepted edit; returns True when the
    user chose to save, False on quit-without-save. The caller owns file I/O.
    """
    try:
        import curses
    except ImportError:  # pragma: no cover - platform-dependent
        raise RuntimeError("config edit requires an interactive terminal") from None
    if not _interactive_terminal():
        raise RuntimeError("config edit requires an interactive terminal")

    import locale

    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:  # pragma: no cover - environment-dependent
        pass  # fall back to the C locale; ACS box chars may degrade

    state = _EditorState(curses, target, merged, user, project, target_path)
    curses.wrapper(state.run)
    return state.saved


def _coerce_option(field: Field, option: str):
    """Keep numeric choice options numeric (e.g. ports) in the saved YAML."""
    if field.kind in ("int", "float"):
        ok, _, value = validate_input(field, option)
        if ok:
            return value
    return option


class _EditorState:
    """Mutable state for one editor session."""

    def __init__(self, curses, target, merged, user, project, target_path):
        self.curses = curses
        self.target = target
        self.merged = merged
        self.user = user
        self.project = project
        self.target_path = target_path
        self.saved = False
        self.sources = resolve_field_sources(user, project)
        self.rows = self._build_rows()
        self.cursor = 0  # index into self.rows
        self.offset = 0
        self.status = ""

    def _refresh_layers(self) -> None:
        """Recompute the effective config and source tags after an edit.

        Re-merging defaults with the user and project layers keeps the
        displayed value and [source] tag in sync with accepted edits; the
        target dict is one of those layers and is mutated in place.
        """
        merged = copy.deepcopy(DEFAULTS)
        for layer in (self.user, self.project):
            merged = utils.merge_configs(merged, layer)
        self.merged = merged
        self.sources = resolve_field_sources(self.user, self.project)

    def _group_prefix(self, group: str) -> str | None:
        """Return the config namespace a field group owns, if any."""
        fields = FIELD_GROUPS.get(group)
        return fields[0].key.split(".")[0] if fields else None

    def _build_rows(self):
        rows = []
        for group, fields in FIELD_GROUPS.items():
            rows.append(("group", group, None))
            for field in fields:
                rows.append(("field", field.label, field))
            prefix = group.split()[0].lower()
            for spec in COLLECTIONS:
                if spec.prefix.split(".", 1)[0] == prefix:
                    rows.append(("collection", spec.title, spec))
            group_prefix = self._group_prefix(group)
            for key in self._custom_keys():
                if key.split(".")[0] == group_prefix:
                    rows.append(("custom", key[len(group_prefix) + 1 :], key))
        known_prefixes = {self._group_prefix(group) for group in FIELD_GROUPS}
        other = [key for key in self._custom_keys() if key.split(".")[0] not in known_prefixes]
        if other:
            # Keys in namespaces no field group owns; `a` there accepts any
            # dotted key.
            rows.append(("group", "Other", None))
            for key in other:
                rows.append(("custom", key, key))
        return rows

    def _custom_keys(self) -> list[str]:
        """List dotted keys present in the file but not managed by the editor."""
        managed = {f.key for f in ALL_FIELDS} | {spec.prefix for spec in COLLECTIONS}

        def has_managed_descendant(prefix: str, value) -> bool:
            for key, child in value.items():
                dotted = f"{prefix}.{key}" if prefix else key
                if dotted in managed or (
                    isinstance(child, dict) and has_managed_descendant(dotted, child)
                ):
                    return True
            return False

        def walk(prefix: str, value) -> None:
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                dotted = f"{prefix}.{key}" if prefix else key
                if dotted in managed:
                    # Managed prefixes hide their children; those keys are
                    # edited through the registry rows or collection modals.
                    continue
                if isinstance(child, dict) and has_managed_descendant(dotted, child):
                    # A mixed mapping (custom keys alongside managed ones):
                    # show its custom children directly, not the container.
                    walk(dotted, child)
                    continue
                keys.append(dotted)
                walk(dotted, child)

        keys: list[str] = []
        walk("", self.target)
        return keys

    # -- rendering --

    def run(self, stdscr) -> None:
        curses = self.curses
        self._init_colors()
        curses.curs_set(0)
        stdscr.keypad(True)
        help_visible = False
        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()
            self._draw(stdscr, height, width, help_visible)
            stdscr.refresh()
            key = stdscr.getch()
            self.status = ""
            if help_visible and key != curses.KEY_RESIZE:
                help_visible = False
                continue
            if key in (curses.KEY_UP, ord("k")):
                self._move(-1)
            elif key in (curses.KEY_DOWN, ord("j")):
                self._move(1)
            elif key in (curses.KEY_ENTER, 10, 13):
                kind, _, _row = self.rows[self.cursor]
                if kind == "collection":
                    self._edit_collection(stdscr, _row)
                elif kind == "custom":
                    self._edit_custom_key(stdscr, _row)
                else:
                    self._edit_field(stdscr)
            elif key == ord("a"):
                self._add_custom_key(stdscr, self._current_group_prefix())
            elif key == ord("d"):
                kind, _, _row = self.rows[self.cursor]
                if kind == "custom":
                    nested_delete(self.target, _row)
                    self._refresh_layers()
                    self.rows = self._build_rows()
                    self.status = f"{_row} removed from this file"
            elif key == ord("r"):
                kind, _, _row = self.rows[self.cursor]
                if kind == "collection":
                    self.status = "reset applies to fields, not profile lists"
                elif kind == "custom":
                    nested_delete(self.target, _row)
                    self._refresh_layers()
                    self.rows = self._build_rows()
                    self.status = f"{_row} removed from this file"
                else:
                    self._reset_field()
            elif key == ord("s"):
                self.saved = True
                return
            elif key == ord("q"):
                return
            elif key == ord("?"):
                help_visible = True

    def _draw(self, stdscr, height, width, help_visible) -> None:
        curses = self.curses
        header = f"arkai config edit — {self.target_path}"
        stdscr.addstr(0, 0, header[: width - 1])
        note = "Saving rewrites this file; YAML comments are not preserved."
        stdscr.addstr(1, 0, note[: width - 1], curses.A_DIM)

        rows = self.rows
        # Screen rows available for the list: reserve header (2) and footer (1).
        list_height = max(1, height - 3)
        self._scroll(list_height)
        bottom = min(self.offset + list_height, len(rows))
        for y, idx in enumerate(range(self.offset, bottom), start=2):
            row_kind, label, field = rows[idx]
            text = self._row_text(field, label, width)
            attr = self._row_attr(row_kind, field, idx)
            try:
                stdscr.addstr(y, 0, text[: width - 1], attr)
            except curses.error:
                pass  # bottom-right corner write

        footer = FOOTER if not self.status else self.status
        try:
            stdscr.addstr(height - 1, 0, footer[: width - 1], curses.A_DIM)
        except curses.error:
            pass
        if help_visible:
            self._draw_help(stdscr, height, width)

    def _init_colors(self) -> None:
        """Set up per-source colors: project bright white, global bright
        yellow, default bright green. Falls back to bold/normal on terminals
        without color support."""
        curses = self.curses
        self.source_attrs = {"project": curses.A_BOLD, "global": 0, "default": 0}
        self.modal_pair = None
        try:
            if not curses.has_colors():
                return
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = curses.COLOR_BLACK
            # 256-color entries avoid the dim/8-color palette (COLOR_WHITE
            # renders gray, COLOR_YELLOW orange-brown on many terminals).
            pairs = {
                "project": 231,
                "global": 226,
                "default": curses.COLOR_GREEN,
            }
            fallbacks = {
                "project": curses.COLOR_WHITE,
                "global": curses.COLOR_YELLOW,
                "default": curses.COLOR_GREEN,
            }
            for number, (source, foreground) in enumerate(pairs.items(), start=1):
                try:
                    curses.init_pair(number, foreground, background)
                except curses.error:
                    curses.init_pair(number, fallbacks[source], background)
                self.source_attrs[source] = curses.color_pair(number) | curses.A_BOLD
            try:
                curses.init_pair(4, self.MODAL_COLOR, background)
            except curses.error:
                curses.init_pair(4, curses.COLOR_CYAN, background)
            self.modal_pair = 4
        except curses.error:
            self.source_attrs = {"project": curses.A_BOLD, "global": 0, "default": 0}
            self.modal_pair = None

    def _row_attr(self, row_kind, field, idx) -> int:
        curses = self.curses
        if idx == self.cursor:
            return curses.A_REVERSE | curses.A_BOLD
        if row_kind in ("group", "collection"):
            return curses.A_DIM
        if isinstance(field, str):
            source = _layer_source(self.user, self.project, field)
            return self.source_attrs.get(source, 0)
        source = self.sources.get(field.key, "default")
        return self.source_attrs.get(source, 0)

    def _row_text(self, field, label, width: int = 0) -> str:
        if field is None:  # group header row
            if width > len(label) + 2:
                return f"{label} {'─' * (width - len(label) - 2)}"
            return label
        if isinstance(field, CollectionSpec):
            profiles = nested_get(self.merged, field.prefix)
            if isinstance(profiles, dict):
                count = sum(1 for v in profiles.values() if isinstance(v, dict))
            else:
                count = 0
            value = f"{count} profiles" if count else "none"
            return f"{label}  ....  {value}"
        if isinstance(field, str):
            # Custom-key row: field is the dotted key; the value is shown
            # as-is with no registry validation applied.
            value = nested_get(self.merged, field)
            if value is _MISSING or value is None:
                value = UNSET_LABEL
            elif isinstance(value, bool):
                value = "true" if value else "false"
            source = _layer_source(self.user, self.project, field)
            return f"{label}  ....  {value}  [{source}]"
        value = nested_get(self.merged, field.key)
        if value is _MISSING or value is None:
            value = UNSET_LABEL
        elif isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, dict) and field.kind == "api_key":
            value = str(value.get("source", UNSET_LABEL))
        source = self.sources.get(field.key, "default")
        return f"{label}  ....  {value}  [{source}]"

    MODAL_COLOR = 51  # 256-color bright cyan

    def _modal_attr(self) -> int:
        curses = self.curses
        pair = getattr(self, "modal_pair", None)
        return curses.color_pair(pair) | curses.A_BOLD if pair else curses.A_BOLD

    def _draw_unicode_box(self, win) -> None:
        """Draw a bright-cyan Unicode box border, ACS-independent.

        ``win.box()`` relies on terminfo ACS entries, which some terminals
        map incorrectly (rendering letters). Unicode box-drawing characters
        render everywhere UTF-8 is active.
        """
        curses = self.curses
        rows, cols = win.getmaxyx()
        horizontal = "─" * (cols - 1)
        attr = self._modal_attr()
        try:
            win.addstr(0, 0, horizontal, attr)
            win.addstr(rows - 1, 0, horizontal, attr)
            for y in range(1, rows - 1):
                win.addstr(y, 0, "│", attr)
                win.addstr(y, cols - 2, "│", attr)
            win.addstr(0, 0, "┌", attr)
            win.addstr(0, cols - 2, "┐", attr)
            win.addstr(rows - 1, 0, "└", attr)
            win.addstr(rows - 1, cols - 2, "┘", attr)
        except curses.error:
            pass  # clipping at screen edges

    def _begin_modal(self, stdscr) -> None:
        """Clear child-modal leftovers before drawing a modal over stdscr.

        Nested modals (member editor, prompts) write directly to the
        physical screen; when they close, the parent modal would leave
        their pixels on screen because it only refreshes its own window.
        Marking stdscr for a full repaint erases them while keeping its
        own content (the field list) visible around the modal.
        """
        stdscr.redrawwin()
        stdscr.noutrefresh()

    def _draw_help(self, stdscr, height, width) -> None:
        lines = [
            "Help",
            "  ↑/↓ or j/k   move between fields",
            "  Enter        edit the selected field",
            "  a            add a key to the current group",
            "  d            delete the selected custom key",
            "  r            reset to default (delete the key from this file)",
            "  s            save and exit",
            "  Esc          cancel the current edit",
            "",
            "Custom keys are stored as typed (YAML) without validation;",
            "use them for model- or plugin-specific settings.",
            "",
            "Source tags: [default] built-in default, [global] from the user",
            "config file, [project] from .arkai.yaml (highest precedence).",
            "Press any key to close this help.",
        ]
        box_h = len(lines) + 2
        box_w = min(width - 4, max(len(line) for line in lines) + 4)
        top = max(0, (height - box_h) // 2)
        left = max(0, (width - box_w) // 2)
        win = self.curses.newwin(box_h, box_w, top, left)
        self._draw_unicode_box(win)
        for i, line in enumerate(lines, start=1):
            win.addstr(i, 2, line[: box_w - 3])
        win.refresh()

    def _scroll(self, list_height: int) -> None:
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + list_height:
            self.offset = self.cursor - list_height + 1

    def _move(self, delta: int) -> None:
        self.cursor = max(0, min(len(self.rows) - 1, self.cursor + delta))
        while self.rows[self.cursor][0] == "group":
            if delta > 0:
                self.cursor = min(len(self.rows) - 1, self.cursor + 1)
                if self.cursor == len(self.rows) - 1 and self.rows[self.cursor][0] == "group":
                    self.cursor = max(0, self.cursor - 1)
                    break
            else:
                self.cursor = max(0, self.cursor - 1)
                if self.cursor == 0 and self.rows[self.cursor][0] == "group":
                    self.cursor = 1 if len(self.rows) > 1 else 0
                    break

    def _current_group_prefix(self) -> str | None:
        """Return the namespace of the group the cursor is inside.

        Returns None for the "Other" group (any namespace) and when no
        group row precedes the cursor.
        """
        for kind, label, _row in reversed(self.rows[: self.cursor + 1]):
            if kind != "group":
                continue
            return None if label == "Other" else self._group_prefix(label)
        return None

    # -- editing --

    def _add_custom_key(self, stdscr, prefix: str | None = None) -> None:
        """Prompt for a dotted key and a YAML value; stored unvalidated.

        ``prefix`` is the namespace of the group the cursor is in; it
        pre-fills the prompt so keys land in the current group by default.
        Editing the hint away (to reach a namespace no group owns) is
        allowed — only the key shape and registry collisions are checked.
        """
        hint = f"{prefix}." if prefix else ""
        key = self._prompt_text(
            stdscr, f"Key under {prefix}" if prefix else "Custom key (dotted)", hint
        )
        if key is None or key == "":
            return
        error = validate_custom_key(key)
        if error:
            self.status = f"Invalid key: {error}"
            return
        raw = self._prompt_text(stdscr, f"Value for {key} (YAML)", "")
        if raw is None:
            return
        ok, message, value = parse_custom_value(raw)
        if not ok:
            self.status = f"Invalid value: {message}"
            return
        nested_set(self.target, key, value)
        self._refresh_layers()
        self.rows = self._build_rows()
        self.status = f"{key} added"

    def _edit_custom_key(self, stdscr, key: str) -> None:
        """Re-enter the YAML value of an existing custom key."""
        current = nested_get(self.target, key)
        if current is _MISSING or current is None:
            # Inherited from another layer: pre-fill with the effective value.
            current = nested_get(self.merged, key)
        if isinstance(current, (dict, list)):
            initial = yaml.safe_dump(current, default_flow_style=True).strip()
        elif current is _MISSING or current is None:
            initial = ""
        else:
            initial = "true" if current is True else "false" if current is False else str(current)
        raw = self._prompt_text(stdscr, f"Value for {key} (YAML)", initial, key_hint=key)
        if raw is None:
            return
        if raw == "":
            nested_delete(self.target, key)
            self._refresh_layers()
            self.rows = self._build_rows()
            self.status = f"{key} removed from this file"
            return
        ok, message, value = parse_custom_value(raw)
        if not ok:
            self.status = f"Invalid value: {message}"
            return
        nested_set(self.target, key, value)
        self._refresh_layers()
        self.status = f"{key} updated"

    def _edit_field(self, stdscr) -> None:
        _, _, field = self.rows[self.cursor]
        if field is None:
            return
        self.status = ""
        if field.kind == "api_key":
            self._edit_api_key(stdscr, field.key, field.label)
        elif field.kind in ("choice", "bool"):
            self._edit_choice(stdscr, field)
        else:
            self._edit_text(stdscr, field)

    def _choice_options(self, field) -> list[str]:
        options: list[str] = []
        if allows_unset(field):
            options.append(UNSET_LABEL)
        if field.kind == "bool":
            options.extend(("true", "false"))
        else:
            options.extend(field_choices(field))
        return options

    def _edit_choice(self, stdscr, field) -> None:
        curses = self.curses
        options = self._choice_options(field)
        selected = 0
        current = nested_get(self.merged, field.key)
        if current is not _MISSING and current is not None:
            current_label = (
                "true" if current is True else "false" if current is False else str(current)
            )
            if current_label in options:
                selected = options.index(current_label)
        box_h = min(len(options) + 4, stdscr.getmaxyx()[0] - 2)
        box_w = min(48, stdscr.getmaxyx()[1] - 2)
        top = max(1, (stdscr.getmaxyx()[0] - box_h) // 2)
        left = max(0, (stdscr.getmaxyx()[1] - box_w) // 2)
        win = curses.newwin(box_h, box_w, top, left)
        win.keypad(True)
        offset = 0
        note = self.status
        while True:
            win.erase()
            self._draw_unicode_box(win)
            title = field.label
            win.addstr(0, 2, title[: box_w - 3], self._modal_attr() | curses.A_REVERSE)
            visible = box_h - 4
            if note:
                line = note[: box_w - 4]
                win.addstr(box_h - 3, 1, line[: box_w - 2], curses.A_DIM)
                visible -= 1
            if selected < offset:
                offset = selected
            elif selected >= offset + visible:
                offset = selected - visible + 1
            for i, opt in enumerate(options[offset : offset + visible], start=1):
                attr = curses.A_REVERSE if offset + i - 1 == selected else 0
                win.addstr(i, 2, opt[: box_w - 3], attr)
            footer = "↑/↓ select · Enter apply · Esc cancel"
            win.addstr(box_h - 1, 2, footer[: box_w - 3], curses.A_DIM)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = min(len(options) - 1, selected + 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                self._apply_choice(field, options[selected])
                return
            elif key == 27:  # Esc
                return
            elif key == curses.KEY_RESIZE:
                continue

    def _apply_choice(self, field, option: str) -> None:
        if option == UNSET_LABEL:
            nested_delete(self.target, field.key)
        elif field.kind == "bool":
            nested_set(self.target, field.key, option == "true")
        else:
            nested_set(self.target, field.key, _coerce_option(field, option))
        self._refresh_layers()

    def _apply_profile_create(self, spec: CollectionSpec, name: str, members: dict) -> None:
        nested_set(self.target, f"{spec.prefix}.{name}", members)
        self._refresh_layers()

    def _apply_profile_delete(self, spec: CollectionSpec, name: str) -> None:
        nested_delete(self.target, f"{spec.prefix}.{name}")
        self._refresh_layers()

    def _prompt_text(
        self, stdscr, title: str, initial: str, note: str = "", key_hint: str = ""
    ) -> str | None:
        """Single-line input modal; returns the buffer, or None on Esc."""
        curses = self.curses
        buffer = initial
        height, width = stdscr.getmaxyx()
        box_h = 6 if not note else 7
        box_w = min(64, width - 2)
        top = max(1, (height - box_h) // 2)
        left = max(0, (width - box_w) // 2)
        win = curses.newwin(box_h, box_w, top, left)
        win.keypad(True)
        curses.curs_set(1)
        try:
            while True:
                win.erase()
                self._draw_unicode_box(win)
                win.addstr(0, 2, title[: box_w - 3], self._modal_attr() | curses.A_REVERSE)
                if note:
                    win.addstr(1, 1, note[: box_w - 2], curses.A_DIM)
                if key_hint:
                    win.addstr(2, 2, f"{key_hint}:", curses.A_DIM)
                win.addstr(3, 2, buffer[: box_w - 3])
                win.addstr(box_h - 1, 2, "Enter apply · Esc cancel"[: box_w - 3], curses.A_DIM)
                win.refresh()
                key = win.getch()
                if key in (curses.KEY_ENTER, 10, 13):
                    return buffer
                if key == 27:  # Esc
                    return None
                if key in (curses.KEY_BACKSPACE, 127, 8):
                    buffer = buffer[:-1]
                elif key == curses.KEY_RESIZE:
                    continue
                elif 32 <= key < 256:
                    buffer += chr(key)
        finally:
            curses.curs_set(0)

    def _edit_text(self, stdscr, field) -> None:
        current = nested_get(self.merged, field.key)
        if current is _MISSING or current is None:
            buffer = ""
        else:
            buffer = "true" if current is True else "false" if current is False else str(current)
        buffer = self._prompt_text(
            stdscr, field.label, buffer, note=self.status, key_hint=field.key
        )
        if buffer is not None:
            self._apply_text(field, buffer)

    def _apply_text(self, field, buffer: str) -> None:
        if field.kind in ("str", "protocol", "url"):
            if buffer == "":
                nested_delete(self.target, field.key)
            else:
                ok, message, value = validate_provider_field(field, buffer)
                if not ok:
                    self.status = f"Invalid value: {message}"
                    return
                nested_set(self.target, field.key, value)
            self._refresh_layers()
            return
        if buffer == "":
            self.status = "Edit cancelled"
            return
        ok, message, value = validate_input(field, buffer)
        if not ok:
            self.status = f"Invalid value: {message}"
            return
        nested_set(self.target, field.key, value)
        self._refresh_layers()

    def _reset_field(self) -> None:
        kind, _, field = self.rows[self.cursor]
        if field is None:
            return
        nested_delete(self.target, field.key)
        self._refresh_layers()
        self.status = f"{field.key} reset; effective value falls back to the next layer"

    # -- collection editors --

    def _collection_profiles(self, spec: CollectionSpec) -> list[tuple[str, dict]]:
        """Return (name, profile) pairs for dict entries, in file order."""
        profiles = nested_get(self.merged, spec.prefix)
        if not isinstance(profiles, dict):
            return []
        return [(name, p) for name, p in profiles.items() if isinstance(p, dict)]

    def _target_profile_names(self, spec: CollectionSpec) -> list[str]:
        profiles = nested_get(self.target, spec.prefix)
        if not isinstance(profiles, dict):
            return []
        return list(profiles.keys())

    def _profile_summary(self, spec: CollectionSpec, profile: dict) -> str:
        if spec.prefix == "inference.profiles":
            model = profile.get("model")
            return str(model) if model is not None else "-"
        if spec.prefix == "agent.cloud.custom.providers":
            model = profile.get("default_model")
            return str(model) if model is not None else "-"
        return f"{profile.get('cpus', '-')} cpus / {profile.get('memory_mb', '-')} MB"

    def _edit_collection(self, stdscr, spec: CollectionSpec) -> None:
        curses = self.curses
        cursor = 0
        note = ""
        while True:
            profiles = self._collection_profiles(spec)
            cursor = max(0, min(cursor, max(0, len(profiles) - 1)))
            height, width = stdscr.getmaxyx()
            box_h = min(max(len(profiles), 1) + 6, height - 2)
            box_w = min(64, width - 2)
            top = max(1, (height - box_h) // 2)
            left = max(0, (width - box_w) // 2)
            win = curses.newwin(box_h, box_w, top, left)
            self._begin_modal(stdscr)
            win.keypad(True)
            win.erase()
            self._draw_unicode_box(win)
            win.addstr(0, 2, spec.title[: box_w - 3], self._modal_attr() | curses.A_REVERSE)
            visible = box_h - 5
            offset = 0
            if cursor >= offset + visible:
                offset = cursor - visible + 1
            if not profiles:
                win.addstr(1, 2, "(no profiles)"[: box_w - 3], curses.A_DIM)
            else:
                for i, (name, profile) in enumerate(profiles[offset : offset + visible]):
                    text = f"{name}  ....  {self._profile_summary(spec, profile)}"
                    attr = curses.A_REVERSE | curses.A_BOLD if offset + i == cursor else 0
                    win.addstr(i + 1, 2, text[: box_w - 3], attr)
            if note:
                win.addstr(box_h - 3, 1, note[: box_w - 2], curses.A_DIM)
            footer = "↑/↓ move · Enter edit · a add · d delete · Esc back"
            win.addstr(box_h - 1, 2, footer[: box_w - 3], curses.A_DIM)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(max(0, len(profiles) - 1), cursor + 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                if profiles:
                    self._edit_profile_members(stdscr, spec, profiles[cursor][0])
                    self._refresh_layers()
            elif key == ord("a"):
                added = self._prompt_add_profile(stdscr, spec)
                if added is not None:
                    name, _ = added
                    self._edit_profile_members(stdscr, spec, name)
                    self._refresh_layers()
                    missing = self._missing_required(spec, name)
                    if missing:
                        self._apply_profile_delete(spec, name)
                        note = f"profile {name} discarded: {', '.join(missing)} is required"
                    elif not nested_get(self.target, f"{spec.prefix}.{name}"):
                        # Left the member editor without changing anything:
                        # don't keep an empty profile in the file.
                        self._apply_profile_delete(spec, name)
            elif key == ord("d"):
                if not profiles:
                    continue
                name = profiles[cursor][0]
                confirm = self._prompt_text(
                    stdscr, "Delete profile", "", note=f"type y to delete {name}"
                )
                if confirm == "y":
                    self._apply_profile_delete(spec, name)
                    if any(n == name for n, _ in self._collection_profiles(spec)):
                        note = (
                            "deleted from this file; still defined in another configuration layer"
                        )
                    else:
                        note = f"profile {name} deleted"
            elif key == 27:  # Esc
                return
            elif key == curses.KEY_RESIZE:
                continue

    def _missing_required(self, spec: CollectionSpec, name: str) -> list[str]:
        """Required member keys absent (or empty) from the profile in target."""
        profile = nested_get(self.target, f"{spec.prefix}.{name}")
        if not isinstance(profile, dict):
            return list(spec.required)
        return [k for k in spec.required if not profile.get(k)]

    def _prompt_add_profile(self, stdscr, spec: CollectionSpec) -> tuple[str, bool] | None:
        """Prompt for a new profile name and create it; None on abort.

        Returns ``(name, created_now)``. The caller opens the full member
        editor so every option is visible; profiles created now are
        discarded when required members are still missing on exit.
        """
        while True:
            name = self._prompt_text(
                stdscr,
                "Provider ID" if spec.prefix == "agent.cloud.custom.providers" else "Profile name",
                "",
            )
            if name is None or name == "":
                return None
            if spec.prefix == "agent.cloud.custom.providers":
                error = validate_provider_id(self._target_profile_names(spec), name)
            else:
                error = validate_profile_name(self._target_profile_names(spec), name)
            if error:
                self.status = f"Invalid name: {error}"
                continue
            break
        self._apply_profile_create(spec, name, {})
        return name, True

    def _edit_profile_members(self, stdscr, spec: CollectionSpec, name: str) -> None:
        curses = self.curses
        members = list(spec.members)
        extra = ()
        if spec.prefix == "sandbox.profiles":
            extra = ("volume", "environment")
        elif spec.prefix == "agent.cloud.custom.providers":
            extra = ("api_key",)
        cursor = 0
        note = ""
        while True:
            total = len(members) + len(extra)
            cursor = max(0, min(cursor, total - 1))
            height, width = stdscr.getmaxyx()
            box_h = min(total + 6, height - 2)
            box_w = min(64, width - 2)
            top = max(1, (height - box_h) // 2)
            left = max(0, (width - box_w) // 2)
            win = curses.newwin(box_h, box_w, top, left)
            self._begin_modal(stdscr)
            win.keypad(True)
            win.erase()
            self._draw_unicode_box(win)
            win.addstr(
                0,
                2,
                f"{spec.title} — {name}"[: box_w - 3],
                self._modal_attr() | curses.A_REVERSE,
            )
            rows: list[tuple[str, str, str, str]] = []  # (kind, member_key, text, tag)
            for member in members:
                dotted = f"{spec.prefix}.{name}.{member.key}"
                text, tag = self._member_row_text(spec, name, member)
                rows.append(("member", member.key, f"{member.label}  ....  {text}  [{tag}]", tag))
            for extra_key in extra:
                dotted = f"{spec.prefix}.{name}.{extra_key}"
                text, tag = self._extra_row_text(spec, name, extra_key)
                label = extra_key.capitalize()
                rows.append((extra_key, extra_key, f"{label}  ....  {text}  [{tag}]", tag))
            visible = box_h - 5
            offset = max(0, min(cursor - visible + 1, max(0, total - visible)))
            for i, (_, _, text, tag) in enumerate(rows[offset : offset + visible]):
                attr = (
                    curses.A_REVERSE | curses.A_BOLD
                    if offset + i == cursor
                    else self.source_attrs.get(tag, 0)
                )
                win.addstr(i + 1, 2, text[: box_w - 3], attr)
            missing = self._missing_required(spec, name)
            if missing:
                note = note or (f"required: {', '.join(missing)} — profile is discarded on exit")
            if note:
                win.addstr(box_h - 3, 1, note[: box_w - 2], curses.A_DIM)
            footer = "↑/↓ move · Enter edit · r reset · Esc back"
            win.addstr(box_h - 1, 2, footer[: box_w - 3], curses.A_DIM)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(total - 1, cursor + 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                kind, member_key, _, _tag = rows[cursor]
                if kind in ("volume", "environment"):
                    self._edit_extra(stdscr, spec, name, kind)
                elif kind == "api_key":
                    self._edit_api_key(
                        stdscr, f"{spec.prefix}.{name}.api_key", f"{spec.title} — {name}"
                    )
                else:
                    self._edit_profile_member(stdscr, spec, name, member_key)
            elif key == ord("r"):
                kind, member_key, _, _tag = rows[cursor]
                dotted = f"{spec.prefix}.{name}.{member_key}"
                nested_delete(self.target, dotted)
                self._refresh_layers()
                note = f"{member_key} reset; effective value falls back to the next layer"
            elif key == 27:  # Esc
                return
            elif key == curses.KEY_RESIZE:
                continue

    def _member_row_text(self, spec: CollectionSpec, name: str, member: Field):
        """Return (display value, source tag) for a scalar member row."""
        dotted = f"{spec.prefix}.{name}.{member.key}"
        value = nested_get(self.merged, dotted)
        if value is not _MISSING and value is not None:
            if isinstance(value, bool):
                value = "true" if value else "false"
            return str(value), _layer_source(self.user, self.project, dotted)
        # Absent: sandbox members inherit the root sandbox value, and cloud
        # provider members inherit the agent.cloud.custom root defaults.
        root_prefix = {
            "sandbox.profiles": "sandbox",
            "agent.cloud.custom.providers": "agent.cloud.custom",
        }.get(spec.prefix)
        if root_prefix:
            root = nested_get(self.merged, f"{root_prefix}.{member.key}")
            if root is not _MISSING and root is not None:
                return str(root), "default"
        return UNSET_LABEL, _layer_source(self.user, self.project, dotted)

    def _extra_row_text(self, spec: CollectionSpec, name: str, extra_key: str):
        """Return (display value, source tag) for extra collection rows."""
        dotted = f"{spec.prefix}.{name}.{extra_key}"
        value = nested_get(self.merged, dotted)
        tag = _layer_source(self.user, self.project, dotted)
        if extra_key == "api_key":
            if value is _MISSING or value is None:
                root = nested_get(self.merged, "agent.cloud.custom.api_key")
                if root is not _MISSING and root is not None:
                    return f"{root['source']}", "default"
                return UNSET_LABEL, tag
            return str(value["source"]), tag
        if value is _MISSING or value is None:
            root = nested_get(self.merged, f"sandbox.{extra_key}")
            if root is not _MISSING and root is not None:
                value, tag = root, "default"
            else:
                value = []
        if extra_key == "volume":
            n = len(value) if isinstance(value, list) else 0
            return (f"{n} entries" if n else "none"), tag
        n = len(value) if isinstance(value, dict) else 0
        return (f"{n} variables" if n else "none"), tag

    def _edit_profile_member(self, stdscr, spec: CollectionSpec, name: str, member_key: str):
        member = next(m for m in spec.members if m.key == member_key)
        dotted = f"{spec.prefix}.{name}.{member_key}"
        # Pre-fill with the profile's own value from target; empty when inherited.
        current = nested_get(self.target, dotted)
        initial = "" if current is _MISSING or current is None else str(current)
        raw = self._prompt_text(stdscr, member.label, initial, key_hint=dotted)
        if raw is None:
            return
        if raw == "":
            if member.kind in ("str", "protocol", "url"):
                nested_delete(self.target, dotted)
                self._refresh_layers()
            return
        ok, message, value = validate_provider_field(member, raw)
        if not ok:
            self.status = f"Invalid value: {message}"
            return
        nested_set(self.target, dotted, value)
        self._refresh_layers()

    def _edit_extra(self, stdscr, spec: CollectionSpec, name: str, kind: str) -> None:
        dotted = f"{spec.prefix}.{name}.{kind}"
        if kind == "volume":
            self._edit_volume_list(stdscr, dotted)
        else:
            self._edit_environment(stdscr, dotted)

    def _edit_api_key(self, stdscr, dotted: str, title: str) -> None:
        """Edit an api_key source declaration at a dotted key."""
        curses = self.curses
        cursor = 0
        note = ""
        sources = ("environment", "keychain", UNSET_LABEL)
        while True:
            current = nested_get(self.target, dotted)
            if not isinstance(current, dict):
                current = nested_get(self.merged, dotted)
            current_source = current.get("source") if isinstance(current, dict) else None
            cursor = max(0, min(cursor, len(sources) - 1))
            height, width = stdscr.getmaxyx()
            box_h = min(len(sources) + 6, height - 2)
            box_w = min(64, width - 2)
            top = max(1, (height - box_h) // 2)
            left = max(0, (width - box_w) // 2)
            win = curses.newwin(box_h, box_w, top, left)
            self._begin_modal(stdscr)
            win.keypad(True)
            win.erase()
            self._draw_unicode_box(win)
            win.addstr(
                0, 2, f"API key — {title}"[: box_w - 3], self._modal_attr() | curses.A_REVERSE
            )
            for i, source in enumerate(sources):
                attr = curses.A_REVERSE | curses.A_BOLD if i == cursor else 0
                marker = " (current)" if source == current_source else ""
                win.addstr(i + 1, 2, f"{source}{marker}"[: box_w - 3], attr)
            if note:
                win.addstr(box_h - 3, 1, note[: box_w - 2], curses.A_DIM)
            footer = "↑/↓ move · Enter select · Esc back"
            win.addstr(box_h - 1, 2, footer[: box_w - 3], curses.A_DIM)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(len(sources) - 1, cursor + 1)
            elif key in (curses.KEY_ENTER, 10, 13):
                source = sources[cursor]
                if source == UNSET_LABEL:
                    nested_delete(self.target, dotted)
                    self._refresh_layers()
                    return
                declaration = self._prompt_api_key_source(stdscr, source, current)
                if declaration is None:
                    note = f"invalid or cancelled {source} declaration"
                    continue
                nested_set(self.target, dotted, declaration)
                self._refresh_layers()
                return
            elif key == 27:  # Esc
                return
            elif key == curses.KEY_RESIZE:
                continue

    def _prompt_api_key_source(self, stdscr, source: str, current: dict | None) -> dict | None:
        """Prompt for the fields of one api_key source; None on abort."""
        declaration: dict = {"source": source}
        if source == "environment":
            initial = ""
            if isinstance(current, dict) and current.get("source") == "environment":
                initial = str(current.get("name", ""))
            raw = self._prompt_text(stdscr, "Environment variable name", initial)
            if raw is None or raw == "":
                return None
            from arkai.cloud_provider import _ENVIRONMENT_NAME_PATTERN

            if not _ENVIRONMENT_NAME_PATTERN.fullmatch(raw):
                return None
            declaration["name"] = raw
        else:
            service = self._prompt_text(
                stdscr,
                "Keychain service",
                str(current.get("service", "")) if isinstance(current, dict) else "",
            )
            if service is None or service == "":
                return None
            account = self._prompt_text(
                stdscr,
                "Keychain account",
                str(current.get("account", "")) if isinstance(current, dict) else "",
            )
            if account is None or account == "":
                return None
            declaration["service"] = service
            declaration["account"] = account
        return declaration

    def _edit_volume_list(self, stdscr, dotted: str) -> None:
        curses = self.curses
        cursor = 0
        note = ""
        while True:
            entries = nested_get(self.target, dotted)
            if not isinstance(entries, list):
                entries = []
            cursor = max(0, min(cursor, max(0, len(entries) - 1)))
            height, width = stdscr.getmaxyx()
            box_h = min(max(len(entries), 1) + 6, height - 2)
            box_w = min(64, width - 2)
            top = max(1, (height - box_h) // 2)
            left = max(0, (width - box_w) // 2)
            win = curses.newwin(box_h, box_w, top, left)
            self._begin_modal(stdscr)
            win.keypad(True)
            win.erase()
            self._draw_unicode_box(win)
            win.addstr(0, 2, "Volumes"[: box_w - 3], self._modal_attr() | curses.A_REVERSE)
            visible = box_h - 5
            offset = max(0, min(cursor - visible + 1, max(0, len(entries) - visible)))
            if not entries:
                win.addstr(1, 2, "(no volumes)"[: box_w - 3], curses.A_DIM)
            else:
                for i, entry in enumerate(entries[offset : offset + visible]):
                    attr = curses.A_REVERSE | curses.A_BOLD if offset + i == cursor else 0
                    win.addstr(i + 1, 2, str(entry)[: box_w - 3], attr)
            if note:
                win.addstr(box_h - 3, 1, note[: box_w - 2], curses.A_DIM)
            footer = "↑/↓ move · a add · d delete · Esc back"
            win.addstr(box_h - 1, 2, footer[: box_w - 3], curses.A_DIM)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(max(0, len(entries) - 1), cursor + 1)
            elif key == ord("a"):
                entry = self._prompt_text(stdscr, "Volume (/path or ~/path, optional :ro)", "")
                if entry is None or entry == "":
                    continue
                error = validate_volume_entry(entries, entry)
                if error:
                    note = f"Invalid volume: {error}"
                    continue
                entries.append(entry)
                nested_set(self.target, dotted, entries)
                self._refresh_layers()
            elif key == ord("d"):
                if not entries:
                    continue
                del entries[cursor]
                nested_set(self.target, dotted, entries)
                self._refresh_layers()
                cursor = min(cursor, max(0, len(entries) - 1))
            elif key == 27:  # Esc
                return
            elif key == curses.KEY_RESIZE:
                continue

    def _edit_environment(self, stdscr, dotted: str) -> None:
        curses = self.curses
        cursor = 0
        note = ""
        while True:
            env = nested_get(self.target, dotted)
            if not isinstance(env, dict):
                env = {}
            keys = list(env.keys())
            cursor = max(0, min(cursor, max(0, len(keys) - 1)))
            height, width = stdscr.getmaxyx()
            box_h = min(max(len(keys), 1) + 6, height - 2)
            box_w = min(64, width - 2)
            top = max(1, (height - box_h) // 2)
            left = max(0, (width - box_w) // 2)
            win = curses.newwin(box_h, box_w, top, left)
            self._begin_modal(stdscr)
            win.keypad(True)
            win.erase()
            self._draw_unicode_box(win)
            win.addstr(0, 2, "Environment"[: box_w - 3], self._modal_attr() | curses.A_REVERSE)
            visible = box_h - 5
            offset = max(0, min(cursor - visible + 1, max(0, len(keys) - visible)))
            if not keys:
                win.addstr(1, 2, "(no variables)"[: box_w - 3], curses.A_DIM)
            else:
                for i, k in enumerate(keys[offset : offset + visible]):
                    text = f"{k} = {env[k]}"
                    attr = curses.A_REVERSE | curses.A_BOLD if offset + i == cursor else 0
                    win.addstr(i + 1, 2, text[: box_w - 3], attr)
            if note:
                win.addstr(box_h - 3, 1, note[: box_w - 2], curses.A_DIM)
            footer = "↑/↓ move · a add · d delete · Esc back"
            win.addstr(box_h - 1, 2, footer[: box_w - 3], curses.A_DIM)
            win.refresh()
            key = win.getch()
            if key in (curses.KEY_UP, ord("k")):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                cursor = min(max(0, len(keys) - 1), cursor + 1)
            elif key == ord("a"):
                raw = self._prompt_text(stdscr, "KEY=VALUE", "")
                if raw is None or raw == "":
                    continue
                ok, message, k, value = validate_env_entry(raw)
                if not ok:
                    note = f"Invalid entry: {message}"
                    continue
                env[k] = value
                nested_set(self.target, dotted, env)
                self._refresh_layers()
                note = f"{k} updated" if k in keys else f"{k} added"
            elif key == ord("d"):
                if not keys:
                    continue
                k = keys[cursor]
                del env[k]
                if env:
                    nested_set(self.target, dotted, env)
                else:
                    nested_delete(self.target, dotted)
                self._refresh_layers()
                cursor = min(cursor, max(0, len(keys) - 2))
            elif key == 27:  # Esc
                return
            elif key == curses.KEY_RESIZE:
                continue
