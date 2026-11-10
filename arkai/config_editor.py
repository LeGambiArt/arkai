"""Curses field editor for configuration files.

Data-driven scalar-field registry plus pure, UI-free helpers so all logic is
unit-testable; the curses UI is isolated in ``run_editor``. Complex mappings
(profiles, volumes, plugin lists) are never touched by the editor: only
registry keys are mutated in the loaded dict.
"""

import copy
import dataclasses
from typing import Any

from arkai import utils
from arkai.config import DEFAULTS

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


def resolve_field_sources(user: dict, project: dict) -> dict[str, str]:
    """Map every registry key to its effective-value source layer."""
    return {
        field.key: (
            "project"
            if key_present(project, field.key)
            else "global"
            if key_present(user, field.key)
            else "default"
        )
        for field in ALL_FIELDS
    }


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

FOOTER = "↑/↓ move · Enter edit · r reset to default · s save · q quit · ? help"


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

    def _build_rows(self):
        rows = []
        for group, fields in FIELD_GROUPS.items():
            rows.append(("group", group, None))
            for field in fields:
                rows.append(("field", field.label, field))
        return rows

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
                self._edit_field(stdscr)
            elif key == ord("r"):
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
            text = self._row_text(field, label)
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
        if row_kind == "group":
            return curses.A_DIM
        source = self.sources.get(field.key, "default")
        return self.source_attrs.get(source, 0)

    def _row_text(self, field, label) -> str:
        if field is None:
            return label
        value = nested_get(self.merged, field.key)
        if value is _MISSING or value is None:
            value = UNSET_LABEL
        elif isinstance(value, bool):
            value = "true" if value else "false"
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

    def _draw_help(self, stdscr, height, width) -> None:
        lines = [
            "Help",
            "  ↑/↓ or j/k   move between fields",
            "  Enter        edit the selected field",
            "  r            reset to default (delete the key from this file)",
            "  s            save and exit",
            "  q            quit without saving",
            "  Esc          cancel the current edit",
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

    # -- editing --

    def _edit_field(self, stdscr) -> None:
        _, _, field = self.rows[self.cursor]
        if field is None:
            return
        self.status = ""
        if field.kind in ("choice", "bool"):
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

    def _edit_text(self, stdscr, field) -> None:
        curses = self.curses
        current = nested_get(self.merged, field.key)
        if current is _MISSING or current is None:
            buffer = ""
        else:
            buffer = "true" if current is True else "false" if current is False else str(current)
        note = self.status
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
                win.addstr(0, 2, field.label[: box_w - 3], self._modal_attr() | curses.A_REVERSE)
                if note:
                    win.addstr(1, 1, note[: box_w - 2], curses.A_DIM)
                win.addstr(2, 2, f"{field.key}:", curses.A_DIM)
                win.addstr(3, 2, buffer[: box_w - 3])
                win.addstr(box_h - 1, 2, "Enter apply · Esc cancel"[: box_w - 3], curses.A_DIM)
                win.refresh()
                key = win.getch()
                if key in (curses.KEY_ENTER, 10, 13):
                    self._apply_text(field, buffer)
                    return
                if key == 27:  # Esc
                    return
                if key in (curses.KEY_BACKSPACE, 127, 8):
                    buffer = buffer[:-1]
                elif key == curses.KEY_RESIZE:
                    continue
                elif 32 <= key < 256:
                    buffer += chr(key)
        finally:
            curses.curs_set(0)

    def _apply_text(self, field, buffer: str) -> None:
        if field.kind == "str":
            if buffer == "":
                nested_delete(self.target, field.key)
            else:
                nested_set(self.target, field.key, buffer)
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
