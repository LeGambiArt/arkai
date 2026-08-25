import os
import tempfile
from pathlib import Path
import pytest
from aitool import config, utils


class TestConfigLoading:
    def test_load_user_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "aitool"
        config_dir.mkdir()
        config_file = config_dir / "aitool.yaml"
        config_file.write_text("agent:\n  name: opencode\ninference:\n  port: 8081\n")

        monkeypatch.setattr(utils, "get_config_home", lambda: str(config_dir))
        monkeypatch.chdir(tmp_path)

        cfg = config.load_config(str(tmp_path))
        assert cfg["agent"]["name"] == "opencode"
        assert cfg["inference"]["port"] == 8081

    def test_load_project_config_overrides_user(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "aitool"
        config_dir.mkdir()
        user_config = config_dir / "aitool.yaml"
        user_config.write_text("agent:\n  name: opencode\ninference:\n  port: 8081\n")

        project_config = tmp_path / ".aitool.yaml"
        project_config.write_text("agent:\n  name: crush\n")

        monkeypatch.setattr(utils, "get_config_home", lambda: str(config_dir))
        monkeypatch.chdir(tmp_path)

        cfg = config.load_config(str(tmp_path))
        assert cfg["agent"]["name"] == "crush"
        assert cfg["inference"]["port"] == 8081  # From user config

    def test_load_config_no_user_no_project(self, tmp_path, monkeypatch):
        config_dir = tmp_path / "aitool"
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
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

    def test_validate_invalid_agent_name(self):
        cfg = {
            "agent": {"name": "invalid"},
            "inference": {"port": 8081, "model": "test.gguf"},
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

    def test_validate_both_model_and_hf(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "hf": "repo/model",
            },
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

    def test_validate_neither_model_nor_hf(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {"port": 8081},
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

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
        # Should not raise
        config.validate_config(cfg)

    def test_validate_invalid_inference_port(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 500,  # Too low
                "model": "test.gguf",
            },
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

    def test_validate_invalid_wtmcp_port(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {"port": 8081, "model": "test.gguf"},
            "wtmcp": {"port": 99999},  # Too high
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

    def test_validate_invalid_gpu_layers(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "gpu_layers": "invalid",  # Not an int
            },
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)

    def test_validate_invalid_context_size(self):
        cfg = {
            "agent": {"name": "opencode"},
            "inference": {
                "port": 8081,
                "model": "test.gguf",
                "context_size": -1,  # Must be positive
            },
        }
        with pytest.raises(SystemExit):
            config.validate_config(cfg)


class TestGetConfigValue:
    def test_get_simple_value(self):
        cfg = {"agent": {"name": "opencode"}}
        assert config.get_config_value(cfg, "agent.name") == "opencode"

    def test_get_nested_value(self):
        cfg = {"inference": {"port": 8081}}
        assert config.get_config_value(cfg, "inference.port") == 8081

    def test_get_nonexistent_value(self):
        cfg = {}
        assert (
            config.get_config_value(cfg, "agent.name", default="crush") == "crush"
        )

    def test_get_deeply_nested(self):
        cfg = {"a": {"b": {"c": "value"}}}
        assert config.get_config_value(cfg, "a.b.c") == "value"

    def test_get_partial_path(self):
        cfg = {"a": {"b": "value"}}
        assert config.get_config_value(cfg, "a.b.c", default="default") == "default"
