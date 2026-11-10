import copy
import os
from unittest.mock import patch

import pytest

from arkai import config, config_editor, utils
from arkai.config_editor import (
    _MISSING,
    ALL_FIELDS,
    COLLECTIONS,
    Field,
    _dynamic_choices,
    _layer_source,
    key_present,
    nested_delete,
    nested_get,
    nested_set,
    resolve_field_sources,
    validate_env_entry,
    validate_input,
    validate_profile_name,
    validate_volume_entry,
)


def _field(key: str, kind: str, **kwargs) -> Field:
    return Field(key, key, kind, **kwargs)


class TestNestedHelpers:
    def test_set_get_roundtrip(self):
        d = {}
        nested_set(d, "inference.port", 9000)
        assert d == {"inference": {"port": 9000}}
        assert nested_get(d, "inference.port") == 9000

    def test_get_missing_returns_sentinel(self):
        assert nested_get({}, "a.b.c") is _MISSING
        assert nested_get({"a": {"b": {}}}, "a.b.c") is _MISSING
        assert nested_get({"a": "not-a-dict"}, "a.b") is _MISSING

    def test_key_present(self):
        d = {}
        assert not key_present(d, "a.b")
        nested_set(d, "a.b", 1)
        assert key_present(d, "a.b")

    def test_set_overwrites_non_dict_intermediate(self):
        d = {"a": "scalar"}
        nested_set(d, "a.b.c", 1)
        assert nested_get(d, "a.b.c") == 1

    def test_delete_removes_empty_parents(self):
        d = {}
        nested_set(d, "a.b.c", 1)
        nested_delete(d, "a.b.c")
        assert d == {}

    def test_delete_keeps_parents_with_siblings(self):
        d = {}
        nested_set(d, "a.b.c", 1)
        nested_set(d, "a.b.d", 2)
        nested_delete(d, "a.b.c")
        assert d == {"a": {"b": {"d": 2}}}

    def test_delete_missing_key_is_noop(self):
        d = {"a": {"b": {}}}
        nested_delete(d, "a.x.y")
        assert d == {"a": {"b": {}}}
        nested_delete(d, "a.b.c")
        assert d == {"a": {"b": {}}}


class TestResolveFieldSources:
    def test_project_wins(self):
        user = {"inference": {"port": 8081}}
        project = {"inference": {"port": 9000}}
        sources = resolve_field_sources(user, project)
        assert sources["inference.port"] == "project"

    def test_global_when_only_in_user(self):
        sources = resolve_field_sources({"agent": {"name": "crush"}}, {})
        assert sources["agent.name"] == "global"
        assert sources["inference.port"] == "default"

    def test_default_when_in_neither(self):
        sources = resolve_field_sources({}, {})
        assert sources["inference.port"] == "default"
        assert sources["agent.name"] == "default"


class TestValidateInput:
    def test_int_bounds(self):
        port = _field("inference.port", "int", minimum=1024, maximum=65535)
        assert validate_input(port, "1023") == (False, "must be >= 1024, got 1023", None)
        assert validate_input(port, "65536") == (False, "must be <= 65535, got 65536", None)
        assert validate_input(port, "1024") == (True, "", 1024)
        assert validate_input(port, "abc") == (False, "must be an integer, got 'abc'", None)

    def test_float_minimum(self):
        temperature = _field("inference.temperature", "float", minimum=0)
        assert validate_input(temperature, "-0.1")[0] is False
        assert validate_input(temperature, "0.7") == (True, "", 0.7)

    def test_context_sizes(self):
        context = _field("inference.context_size", "context")
        for raw in ("32768", "32k", "1M"):
            ok, _, value = validate_input(context, raw)
            assert ok, raw
            assert value == raw  # raw string stored; loader accepts suffixes
        for raw in ("invalid", "0", "-100"):
            assert validate_input(context, raw)[0] is False, raw

    def test_bool(self):
        boolean = _field("sandbox.enabled", "bool")
        assert validate_input(boolean, "true") == (True, "", True)
        assert validate_input(boolean, "false") == (True, "", False)
        assert validate_input(boolean, "yes")[0] is False

    def test_choice_membership(self):
        choice = _field("agent.mode", "choice", choices=("local", "cloud"))
        assert validate_input(choice, "cloud") == (True, "", "cloud")
        assert validate_input(choice, "remote")[0] is False

    def test_str_passthrough_and_empty(self):
        string = _field("inference.model", "str")
        assert validate_input(string, "m.gguf") == (True, "", "m.gguf")
        assert validate_input(string, "") == (True, "", "")

    def test_empty_input_on_int_rejected(self):
        assert validate_input(_field("inference.port", "int", minimum=1024), "")[0] is False

    def test_dynamic_choice(self):
        backend = _field("inference.backend", "choice", dynamic="backends")
        ok, _, value = validate_input(backend, "llama-cpp")
        assert ok and value == "llama-cpp"
        assert validate_input(backend, "nope")[0] is False


class TestRegistryInvariants:
    def test_keys_unique(self):
        keys = [f.key for f in ALL_FIELDS]
        assert len(keys) == len(set(keys))

    def test_choice_fields_have_choices(self):
        for f in ALL_FIELDS:
            if f.kind == "choice":
                assert f.choices is not None or f.dynamic is not None, f.key
            else:
                assert f.choices is None and f.dynamic is None, f.key

    def test_dynamic_agents(self):
        choices = _dynamic_choices("agents")
        assert isinstance(choices, tuple)
        assert len(choices) > 0
        assert "opencode" in choices

    def test_dynamic_backends(self, monkeypatch):
        monkeypatch.setattr(
            "arkai.inference_backend.get_backend_names", lambda: {"llama-cpp", "mlx"}
        )
        choices = _dynamic_choices("backends")
        assert choices == ("llama-cpp", "mlx")

    def test_unknown_dynamic_provider(self):
        with pytest.raises(ValueError, match="Unknown dynamic choice provider"):
            _dynamic_choices("nope")

    def test_unset_support_follows_defaults(self):
        cloud = next(f for f in ALL_FIELDS if f.key == "agent.cloud.default")
        assert config_editor.allows_unset(cloud)
        mode = next(f for f in ALL_FIELDS if f.key == "agent.mode")
        assert not config_editor.allows_unset(mode)


def _make_state(target=None, user=None, project=None):
    """Build an _EditorState without curses (fixture for mutation tests)."""
    state = config_editor._EditorState.__new__(config_editor._EditorState)
    state.curses = None
    state.user = user if user is not None else {}
    state.project = project if project is not None else {}
    state.target = target if target is not None else state.project
    state.target_path = ".arkai.yaml"
    state.saved = False
    state.status = ""
    state.cursor = 0
    state.offset = 0
    state.merged = copy.deepcopy(config.DEFAULTS)
    state.sources = resolve_field_sources(state.user, state.project)
    state.rows = state._build_rows()
    return state


class TestEditorStateRefresh:
    def test_apply_choice_updates_effective_value_and_source(self):
        """Applied edits must show in the list (merged) and re-tag sources."""
        state = _make_state()

        port = next(f for f in ALL_FIELDS if f.key == "inference.port")
        state._apply_choice(port, "9000")
        assert nested_get(state.merged, "inference.port") == 9000
        assert state.sources["inference.port"] == "project"

        state._apply_choice(port, config_editor.UNSET_LABEL)
        assert nested_get(state.merged, "inference.port") == config.DEFAULTS["inference"]["port"]
        assert state.sources["inference.port"] == "default"


class TestLayerSource:
    def test_only_project(self):
        assert _layer_source({}, {"a": {"b": 1}}, "a.b") == "project"

    def test_only_user(self):
        assert _layer_source({"a": {"b": 1}}, {}, "a.b") == "global"

    def test_project_wins(self):
        assert _layer_source({"a": {"b": 1}}, {"a": {"b": 2}}, "a.b") == "project"

    def test_neither(self):
        assert _layer_source({}, {}, "a.b") == "default"


class TestCollectionRegistry:
    def test_member_keys_unique_per_spec(self):
        for spec in COLLECTIONS:
            keys = [m.key for m in spec.members]
            assert len(keys) == len(set(keys)), f"duplicate members in {spec.prefix}"

    def test_inference_members_match_model_settings(self):
        inference = next(s for s in COLLECTIONS if s.prefix == "inference.profiles")
        sampling = {m.key for m in inference.members} - {"model"}
        assert sampling == config.MODEL_SETTINGS

    def test_required_subset_of_members(self):
        for spec in COLLECTIONS:
            keys = {m.key for m in spec.members}
            assert set(spec.required) <= keys

    def test_no_scalar_key_inside_collection_namespace(self):
        for spec in COLLECTIONS:
            for field in ALL_FIELDS:
                assert not field.key.startswith(spec.prefix + "."), (
                    f"scalar key {field.key} collides with {spec.prefix}"
                )


class TestValidateProfileName:
    def test_accepts_valid(self):
        assert validate_profile_name([], "dev_1") is None

    @pytest.mark.parametrize("name", ["", "bad name", "default", "active", "Active"])
    def test_rejects_invalid(self, name):
        assert validate_profile_name([], name) is not None

    def test_rejects_duplicate(self):
        assert validate_profile_name(["dev"], "dev") is not None


class TestValidateVolumeEntry:
    def test_accepts_absolute_and_home_ro(self):
        assert validate_volume_entry([], "/a") is None
        assert validate_volume_entry([], "~/b:ro") is None

    def test_rejects_relative(self):
        assert validate_volume_entry([], "relative") is not None

    def test_rejects_bad_flag(self):
        assert validate_volume_entry([], "/a:rw") is not None

    def test_rejects_conflicting_duplicate(self):
        assert validate_volume_entry(["/a"], "/a:ro") is not None


class TestValidateEnvEntry:
    def test_accepts_simple(self):
        assert validate_env_entry("K=v") == (True, "", "K", "v")

    def test_value_may_contain_equals(self):
        assert validate_env_entry("K=v=w") == (True, "", "K", "v=w")

    def test_rejects_missing_separator(self):
        assert validate_env_entry("novalue")[0] is False

    def test_rejects_empty_key(self):
        assert validate_env_entry("=v")[0] is False


class TestProfileStateMutations:
    def test_apply_profile_create(self):
        state = _make_state()
        spec = COLLECTIONS[0]
        state._apply_profile_create(spec, "fast", {"model": "test.gguf"})
        assert nested_get(state.target, "inference.profiles.fast") == {"model": "test.gguf"}
        assert nested_get(state.merged, "inference.profiles.fast.model") == "test.gguf"

    def test_apply_profile_delete_prunes_empty_parent(self):
        state = _make_state()
        spec = COLLECTIONS[1]
        state._apply_profile_create(spec, "dev", {})
        state._apply_profile_delete(spec, "dev")
        # DEFAULTS carries sandbox.profiles: {} so the effective value stays
        # an empty dict; the target file must not retain the pruned chain.
        assert "sandbox" not in state.target
        assert nested_get(state.merged, "sandbox.profiles") == {}

    def test_missing_required_reports_empty_members(self):
        state = _make_state()
        spec = COLLECTIONS[0]  # inference: required ("model",)
        state._apply_profile_create(spec, "fast", {})
        assert state._missing_required(spec, "fast") == ["model"]
        nested_set(state.target, "inference.profiles.fast.model", "m")
        assert state._missing_required(spec, "fast") == []

    def test_missing_required_absent_profile(self):
        state = _make_state()
        assert state._missing_required(COLLECTIONS[0], "nope") == ["model"]

    def test_member_delete_falls_back_to_root_value(self):
        state = _make_state(project={"sandbox": {"memory_mb": 1024}})
        nested_set(state.target, "sandbox.profiles.dev", {"memory_mb": 512})
        nested_delete(state.target, "sandbox.profiles.dev.memory_mb")
        state._refresh_layers()
        # The emptied profile and its pruned parent are gone from the file;
        # the effective value falls back to the root sandbox setting.
        assert "profiles" not in state.target["sandbox"]
        assert nested_get(state.merged, "sandbox.profiles.dev.memory_mb") is _MISSING
        assert nested_get(state.merged, "sandbox.memory_mb") == 1024


def _config_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home / ".config" / "arkai"


class TestCmdConfigEdit:
    def _run(self, args=None):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        config.ingest_cli_options(subparsers)
        parsed = parser.parse_args(["config"] + (args or []))
        config.exec_cmd(parsed)

    def test_edit_global_creates_file(self, tmp_path, monkeypatch):
        config_home = _config_home(tmp_path, monkeypatch)

        def script(target, merged, user, project, target_path):
            nested_set(target, "inference.port", 9000)
            return True

        with patch.object(config_editor, "run_editor", side_effect=script) as run:
            self._run(["edit", "--global"])
        run.assert_called_once()
        path = config_home / "arkai.yaml"
        assert path.exists()
        assert utils.load_yaml(str(path))["inference"]["port"] == 9000

    def test_edit_project_preserves_non_registry_keys(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        project_path = tmp_path / ".arkai.yaml"
        utils.save_yaml(str(project_path), {"inference": {"profiles": {"p": {"model": "m"}}}})

        def script(target, merged, user, project, target_path):
            nested_set(target, "inference.port", 9000)
            return True

        with patch.object(config_editor, "run_editor", side_effect=script):
            self._run(["edit", "--project"])
        saved = utils.load_yaml(str(project_path))
        assert saved["inference"]["port"] == 9000
        assert saved["inference"]["profiles"] == {"p": {"model": "m"}}

    def test_edit_project_without_file_fails(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        with patch.object(config_editor, "run_editor", return_value=True):
            with pytest.raises(RuntimeError, match="No project configuration file"):
                self._run(["edit", "--project"])

    def test_both_flags_mutually_exclusive(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            self._run(["edit", "--global", "--project"])

    def test_default_target_project_when_exists(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".arkai.yaml").write_text("agent: {name: opencode}\n")
        with patch.object(config_editor, "run_editor", return_value=False) as run:
            self._run(["edit"])
        assert run.call_args[0][4] == ".arkai.yaml"

    def test_default_target_global_when_no_project(self, tmp_path, monkeypatch):
        config_home = _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        with patch.object(config_editor, "run_editor", return_value=False) as run:
            self._run(["edit"])
        assert run.call_args[0][4] == os.path.join(str(config_home), "arkai.yaml")

    def test_abort_leaves_file_unchanged(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        project_path = tmp_path / ".arkai.yaml"
        original = "agent: {name: opencode}\n"
        project_path.write_text(original)

        def script(target, merged, user, project, target_path):
            nested_set(target, "inference.port", 9000)  # mutation must not be written
            return False

        with patch.object(config_editor, "run_editor", side_effect=script):
            self._run(["edit"])
        assert project_path.read_text() == original

    def test_edit_saves_inference_profile_alongside_existing_keys(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        project_path = tmp_path / ".arkai.yaml"
        utils.save_yaml(str(project_path), {"inference": {"model": "base.gguf", "custom_key": 1}})

        def script(target, merged, user, project, target_path):
            nested_set(target, "inference.profiles.p1", {"model": "m"})
            return True

        with patch.object(config_editor, "run_editor", side_effect=script):
            self._run(["edit"])
        saved = utils.load_yaml(str(project_path))
        assert saved["inference"]["model"] == "base.gguf"
        assert saved["inference"]["custom_key"] == 1
        assert saved["inference"]["profiles"]["p1"]["model"] == "m"


class TestCmdConfigInit:
    def _run(self, args=None):
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        config.ingest_cli_options(subparsers)
        parsed = parser.parse_args(["config"] + (args or []))
        config.exec_cmd(parsed)

    def test_tui_save_writes_mutated_file(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)

        def script(target, merged, user, project, target_path):
            assert target["sandbox"]["cpus"] == 200  # seed preserved
            nested_set(target, "agent.name", "crush")
            return True

        with patch.object(config_editor, "run_editor", side_effect=script):
            self._run(["init", "--tui"])
        saved = utils.load_yaml(".arkai.yaml")
        assert saved["agent"]["name"] == "crush"
        assert saved["sandbox"]["cpus"] == 200

    def test_tui_abort_writes_no_file(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        with patch.object(config_editor, "run_editor", return_value=False):
            self._run(["init", "--tui"])
        assert not (tmp_path / ".arkai.yaml").exists()

    def test_tui_global_writes_config_home(self, tmp_path, monkeypatch):
        config_home = _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)

        def script(target, merged, user, project, target_path):
            nested_set(target, "agent.name", "crush")
            return True

        with patch.object(config_editor, "run_editor", side_effect=script):
            self._run(["init", "--tui", "--global"])
        path = config_home / "arkai.yaml"
        assert path.exists()
        assert utils.load_yaml(str(path))["agent"]["name"] == "crush"
        assert not (tmp_path / ".arkai.yaml").exists()

    def test_global_without_tui_writes_seed(self, tmp_path, monkeypatch):
        config_home = _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        self._run(["init", "--global"])
        path = config_home / "arkai.yaml"
        assert path.exists()
        saved = utils.load_yaml(str(path))
        assert saved["agent"]["name"] == "opencode"
        assert saved["sandbox"]["cpus"] == 200

    def test_global_project_mutually_exclusive(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            self._run(["init", "--global", "--project"])

    def test_existing_global_target_fails(self, tmp_path, monkeypatch):
        config_home = _config_home(tmp_path, monkeypatch)
        config_home.mkdir(parents=True)
        (config_home / "arkai.yaml").write_text("agent: {name: opencode}\n")
        with pytest.raises(RuntimeError, match="Configuration file already exists"):
            self._run(["init", "--global"])

    def test_existing_project_target_fails(self, tmp_path, monkeypatch):
        _config_home(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".arkai.yaml").write_text("agent: {name: opencode}\n")
        with pytest.raises(RuntimeError, match="Configuration file already exists"):
            self._run(["init"])
