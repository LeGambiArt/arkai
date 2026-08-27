from arkai import config, utils


class TestConfigLoading:
    def test_load_user_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "arkai"
        config_dir.mkdir()
        config_file = config_dir / "arkai.yaml"
        config_file.write_text("agent:\n  name: opencode\ninference:\n  port: 8081\n")

        monkeypatch.setattr(utils, "get_config_home", lambda: str(config_dir))
        monkeypatch.chdir(tmp_path)

        cfg = config.load_config(str(tmp_path))
        assert cfg["agent"]["name"] == "opencode"
        assert cfg["inference"]["port"] == 8081

    def test_load_project_config_overrides_user(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "arkai"
        config_dir.mkdir()
        user_config = config_dir / "arkai.yaml"
        user_config.write_text("agent:\n  name: opencode\ninference:\n  port: 8081\n")

        project_config = tmp_path / ".arkai.yaml"
        project_config.write_text("agent:\n  name: crush\n")

        monkeypatch.setattr(utils, "get_config_home", lambda: str(config_dir))
        monkeypatch.chdir(tmp_path)

        cfg = config.load_config(str(tmp_path))
        assert cfg["agent"]["name"] == "crush"
        assert cfg["inference"]["port"] == 8081  # From user config

    def test_load_config_no_user_no_project(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "arkai"
        config_dir.mkdir()
        monkeypatch.setattr(utils, "get_config_home", lambda: str(config_dir))
        monkeypatch.chdir(tmp_path)

        cfg = config.load_config(str(tmp_path))
        # Should have defaults
        assert "inference" in cfg
        assert "agent" in cfg


class TestConfigValidation:
    def test_validate_missing_agent_name(self):
        cfg = {"agent": {}, "inference": {"port": 8081, "model": "test.gguf"}}
        assert config.validate_config(cfg) is False

    def test_validate_invalid_agent_name(self):
        cfg = {
            "agent": {"name": "invalid"},
            "inference": {"port": 8081, "model": "test.gguf"},
        }
        assert config.validate_config(cfg) is False

    def test_validate_both_model_and_hf(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "hf": "repo/model",
            },
        }
        assert config.validate_config(cfg) is False

    def test_validate_neither_model_nor_hf(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {"port": 8081},
        }
        assert config.validate_config(cfg) is False

    def test_validate_valid_config(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "gpu_layers": -1,
                "context_size": 65536,
            },
        }
        # Should return True
        assert config.validate_config(cfg) is True

    def test_validate_invalid_inference_port(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 500,  # Too low
                "model": "test.gguf",
            },
        }
        assert config.validate_config(cfg) is False

    def test_validate_invalid_wtmcp_port(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {"port": 8081, "model": "test.gguf"},
            "wtmcp": {"port": 99999},  # Too high
        }
        assert config.validate_config(cfg) is False

    def test_validate_invalid_gpu_layers(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "gpu_layers": "invalid",  # Not an int
            },
        }
        assert config.validate_config(cfg) is False

    def test_validate_invalid_context_size(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "context_size": -1,  # Must be positive
            },
        }
        assert config.validate_config(cfg) is False


class TestValidateVolumes:
    """Test volume validation in config."""

    BASE_CFG = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }

    def _cfg_with_sandbox(self, sandbox: dict) -> dict:
        return {**self.BASE_CFG, "sandbox": sandbox}

    def test_valid_volume_path_only(self) -> None:
        """A plain path volume is valid."""
        cfg = self._cfg_with_sandbox({"volume": ["/data"]})
        assert config.validate_config(cfg) is True

    def test_valid_volume_with_ro_flag(self) -> None:
        """A volume with :ro flag is valid."""
        cfg = self._cfg_with_sandbox({"volume": ["/data:ro"]})
        assert config.validate_config(cfg) is True

    def test_invalid_volume_not_absolute(self) -> None:
        """A relative path volume is invalid."""
        cfg = self._cfg_with_sandbox({"volume": ["data/subdir"]})
        assert config.validate_config(cfg) is False

    def test_invalid_volume_bad_flag(self) -> None:
        """An unsupported flag is invalid."""
        cfg = self._cfg_with_sandbox({"volume": ["/data:rw"]})
        assert config.validate_config(cfg) is False

    def test_invalid_volume_conflicting_flags(self) -> None:
        """Same path with different flags is invalid."""
        cfg = self._cfg_with_sandbox({"volume": ["/data", "/data:ro"]})
        assert config.validate_config(cfg) is False

    def test_valid_profile_volumes(self) -> None:
        """Profile volumes are also validated."""
        cfg = self._cfg_with_sandbox({"profiles": {"gpu": {"volume": ["/data:ro", "/logs"]}}})
        assert config.validate_config(cfg) is True

    def test_invalid_profile_volume_conflict(self) -> None:
        """Conflicting volumes in a profile are invalid."""
        cfg = self._cfg_with_sandbox({"profiles": {"gpu": {"volume": ["/data", "/data:ro"]}}})
        assert config.validate_config(cfg) is False


class TestValidateEnvironment:
    """Test environment variable validation in config."""

    BASE_CFG = {
        "agent": {"name": "opencode"},
        "inference": {"model": "test.gguf"},
    }

    def _cfg_with_sandbox(self, sandbox: dict) -> dict:
        return {**self.BASE_CFG, "sandbox": sandbox}

    def test_valid_environment_dict(self) -> None:
        """A valid environment dict passes."""
        cfg = self._cfg_with_sandbox({"environment": {"FOO": "bar", "COUNT": 42}})
        assert config.validate_config(cfg) is True

    def test_invalid_environment_not_dict(self) -> None:
        """environment must be a dict, not a list."""
        cfg = self._cfg_with_sandbox({"environment": ["FOO=bar"]})
        assert config.validate_config(cfg) is False

    def test_invalid_environment_non_scalar_value(self) -> None:
        """environment values must be scalar (not dict or list)."""
        cfg = self._cfg_with_sandbox({"environment": {"FOO": {"nested": "dict"}}})
        assert config.validate_config(cfg) is False

    def test_invalid_environment_list_value(self) -> None:
        """environment values must not be lists."""
        cfg = self._cfg_with_sandbox({"environment": {"FOO": ["a", "b"]}})
        assert config.validate_config(cfg) is False

    def test_valid_profile_environment(self) -> None:
        """Profile environment dict is also validated."""
        cfg = self._cfg_with_sandbox(
            {"profiles": {"gpu": {"environment": {"CUDA_VISIBLE_DEVICES": "0"}}}}
        )
        assert config.validate_config(cfg) is True

    def test_invalid_profile_environment_not_dict(self) -> None:
        """Profile environment must be a dict."""
        cfg = self._cfg_with_sandbox({"profiles": {"gpu": {"environment": "FOO=bar"}}})
        assert config.validate_config(cfg) is False


class TestGetConfigValue:
    def test_get_simple_value(self):
        cfg = {"agent": {"name": "opencode"}}
        assert config.get_config_value(cfg, "agent.name") == "opencode"

    def test_get_nested_value(self):
        cfg = {"inference": {"port": 8081}}
        assert config.get_config_value(cfg, "inference.port") == 8081

    def test_get_nonexistent_value(self):
        cfg = {}
        assert config.get_config_value(cfg, "agent.name", default="crush") == "crush"

    def test_get_deeply_nested(self):
        cfg = {"a": {"b": {"c": "value"}}}
        assert config.get_config_value(cfg, "a.b.c") == "value"

    def test_get_partial_path(self):
        cfg = {"a": {"b": "value"}}
        assert config.get_config_value(cfg, "a.b.c", default="default") == "default"
