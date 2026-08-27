"""Unit tests for sandbox profile utilities."""

from unittest.mock import patch

import pytest

from arkai import agent as agent_module
from arkai import sandbox


class TestProfileValidation:
    """Test profile name validation."""

    def test_valid_identifier(self) -> None:
        """Valid identifiers should pass."""
        assert sandbox._is_valid_profile_name("minimal")
        assert sandbox._is_valid_profile_name("gpu_heavy")
        assert sandbox._is_valid_profile_name("Profile123")
        assert sandbox._is_valid_profile_name("_leading")

    def test_invalid_identifier(self) -> None:
        """Invalid identifiers should fail."""
        assert not sandbox._is_valid_profile_name("")
        assert not sandbox._is_valid_profile_name("gpu-heavy")  # hyphen not allowed
        assert not sandbox._is_valid_profile_name("gpu heavy")  # space not allowed

    def test_reserved_names_rejected(self) -> None:
        """Reserved names 'default' and 'active' should be rejected."""
        assert not sandbox._is_valid_profile_name("default")
        assert not sandbox._is_valid_profile_name("active")
        assert not sandbox._is_valid_profile_name("DEFAULT")
        assert not sandbox._is_valid_profile_name("ACTIVE")


class TestProfileRetrieval:
    """Test reading profiles from config."""

    def test_get_default_profile(self) -> None:
        """Should extract defaults from sandbox root."""
        cfg = {
            "sandbox": {
                "path": "/usr/bin/arapuca",
                "memory_mb": 4096,
                "cpus": 4,
                "pids": 512,
                "timeout": 3600,
            }
        }
        profile = sandbox._get_default_profile(cfg)
        assert profile["path"] == "/usr/bin/arapuca"
        assert profile["memory_mb"] == 4096
        assert profile["cpus"] == 4

    def test_get_default_profile_fallback(self) -> None:
        """Should provide fallback values if keys missing."""
        cfg = {"sandbox": {}}
        profile = sandbox._get_default_profile(cfg)
        assert profile["path"] == "arapuca"
        assert profile["memory_mb"] == 2048
        assert profile["cpus"] == 2

    def test_get_profile_exists(self) -> None:
        """Should retrieve existing profile."""
        cfg = {
            "sandbox": {
                "profiles": {
                    "gpu": {
                        "path": "/custom/arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 256,
                        "timeout": 0,
                    }
                }
            }
        }
        profile = sandbox._get_profile(cfg, "gpu")
        assert profile is not None
        assert profile["path"] == "/custom/arapuca"

    def test_get_profile_not_exists(self) -> None:
        """Should return None for non-existent profile."""
        cfg = {"sandbox": {"profiles": {}}}
        profile = sandbox._get_profile(cfg, "nonexistent")
        assert profile is None

    def test_profile_exists(self) -> None:
        """Should check if profile exists."""
        cfg = {"sandbox": {"profiles": {"gpu": {}, "minimal": {}}}}
        assert sandbox._profile_exists(cfg, "gpu")
        assert not sandbox._profile_exists(cfg, "nonexistent")


class TestProfileResolution:
    """Test sandbox profile resolution in agent."""

    def test_resolve_cli_flag_priority(self) -> None:
        """CLI --sandbox flag should have highest priority."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": "active_prof",
                "profiles": {
                    "cli_prof": {
                        "path": "arapuca",
                        "memory_mb": 1024,
                        "cpus": 1,
                        "pids": 128,
                        "timeout": 0,
                    },
                    "active_prof": {
                        "path": "arapuca",
                        "memory_mb": 4096,
                        "cpus": 4,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        # CLI flag should win
        profile = agent_module._resolve_sandbox_profile(cfg, "cli_prof")
        assert profile["memory_mb"] == 1024

    def test_resolve_active_profile_second(self) -> None:
        """Active profile should be second priority."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": "active_prof",
                "profiles": {
                    "active_prof": {
                        "path": "arapuca",
                        "memory_mb": 4096,
                        "cpus": 4,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        # No CLI flag, should use active_profile
        profile = agent_module._resolve_sandbox_profile(cfg, None)
        assert profile["memory_mb"] == 4096

    def test_resolve_defaults_fallback(self) -> None:
        """Defaults should be last resort."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {},
            }
        }

        # No CLI flag, no active profile
        profile = agent_module._resolve_sandbox_profile(cfg, None)
        assert profile["memory_mb"] == 2048
        assert profile["cpus"] == 2

    def test_resolve_nonexistent_profile(self) -> None:
        """Should raise RuntimeError for non-existent CLI profile."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {},
            }
        }

        with pytest.raises(RuntimeError, match="Sandbox profile not found"):
            agent_module._resolve_sandbox_profile(cfg, "nonexistent")


class TestListCommand:
    """Test sandbox list command output."""

    def test_list_shows_no_profiles(self, capsys: pytest.CaptureFixture) -> None:
        """Should show no profiles message when none exist."""
        from unittest.mock import patch

        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {},
            }
        }

        with patch("arkai.config.load_config", return_value=cfg):
            sandbox.cmd_sandbox_list()

        captured = capsys.readouterr()
        assert "No custom profiles defined" in captured.out

    def test_list_shows_profiles(self, capsys: pytest.CaptureFixture) -> None:
        """Should list all profiles with summary info."""
        from unittest.mock import patch

        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": "gpu",
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 5,
                        "pids": 512,
                        "timeout": 3600,
                    },
                    "minimal": {
                        "path": "arapuca",
                        "memory_mb": 1024,
                        "cpus": 1,
                        "pids": 128,
                        "timeout": 0,
                    },
                },
            }
        }

        with patch("arkai.config.load_config", return_value=cfg):
            sandbox.cmd_sandbox_list()

        captured = capsys.readouterr()
        assert "Active Profile: gpu" in captured.out
        assert "gpu" in captured.out
        assert "minimal" in captured.out
        assert "✓" in captured.out  # Mark active profile


class TestShowCommand:
    """Test sandbox show command output."""

    def test_show_default(self, capsys: pytest.CaptureFixture) -> None:
        """Should show default profile when 'default' is specified."""
        from unittest.mock import patch

        cfg = {
            "sandbox": {
                "path": "/usr/bin/arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {},
            }
        }

        with patch("arkai.config.load_config", return_value=cfg):
            sandbox.cmd_sandbox_show("default")

        captured = capsys.readouterr()
        assert "Default Settings" in captured.out
        assert "path:      /usr/bin/arapuca" in captured.out
        assert "cpus:      2" in captured.out

    def test_show_active_profile(self, capsys: pytest.CaptureFixture) -> None:
        """Should show active profile when 'active' is specified."""
        from unittest.mock import patch

        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": "gpu",
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 5,
                        "pids": 512,
                        "timeout": 3600,
                    },
                },
            }
        }

        with patch("arkai.config.load_config", return_value=cfg):
            sandbox.cmd_sandbox_show("active")

        captured = capsys.readouterr()
        assert "Active Profile: gpu" in captured.out
        assert "memory_mb: 8192" in captured.out
        assert "cpus:      5" in captured.out

    def test_show_active_defaults_when_none_set(self, capsys: pytest.CaptureFixture) -> None:
        """Should show defaults when 'active' but no active profile set."""
        from unittest.mock import patch

        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {},
            }
        }

        with patch("arkai.config.load_config", return_value=cfg):
            sandbox.cmd_sandbox_show("active")

        captured = capsys.readouterr()
        assert "using defaults" in captured.out
        assert "cpus:      2" in captured.out

    def test_show_named_profile(self, capsys: pytest.CaptureFixture) -> None:
        """Should show details for named profile."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {
                    "minimal": {
                        "path": "arapuca",
                        "memory_mb": 1024,
                        "cpus": 1,
                        "pids": 128,
                        "timeout": 0,
                    },
                },
            }
        }

        with patch("arkai.config.load_config", return_value=cfg):
            sandbox.cmd_sandbox_show("minimal")

        captured = capsys.readouterr()
        assert "Profile: minimal" in captured.out
        assert "memory_mb: 1024" in captured.out
        assert "cpus:      1" in captured.out


class TestCreateCommand:
    """Test sandbox create command."""

    def test_create_from_defaults(self, capsys: pytest.CaptureFixture) -> None:
        """Should create profile based on defaults."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {},
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_create("test", cpus=4, memory=4096)

        # Verify config was saved
        mock_save.assert_called_once()
        saved_cfg = mock_save.call_args[0][1]
        assert "test" in saved_cfg["sandbox"]["profiles"]
        assert saved_cfg["sandbox"]["profiles"]["test"]["cpus"] == 4
        assert saved_cfg["sandbox"]["profiles"]["test"]["memory_mb"] == 4096

        captured = capsys.readouterr()
        assert "Created sandbox profile 'test'" in captured.out

    def test_create_from_base(self, capsys: pytest.CaptureFixture) -> None:
        """Should inherit from existing profile."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 3600,
                    }
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_create("gpu_constrained", from_profile="gpu", timeout=1800)

        mock_save.assert_called_once()
        saved_cfg = mock_save.call_args[0][1]
        new_profile = saved_cfg["sandbox"]["profiles"]["gpu_constrained"]
        # Should inherit from gpu
        assert new_profile["memory_mb"] == 8192
        assert new_profile["cpus"] == 8
        # But override timeout
        assert new_profile["timeout"] == 1800

    def test_create_invalid_name(self) -> None:
        """Should reject invalid profile name."""
        cfg = {"sandbox": {"profiles": {}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_create("invalid-name")

    def test_create_reserved_name(self) -> None:
        """Should reject reserved names."""
        cfg = {"sandbox": {"profiles": {}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_create("default")

    def test_create_duplicate(self) -> None:
        """Should reject duplicate profile name."""
        cfg = {"sandbox": {"profiles": {"existing": {}}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_create("existing")

    def test_create_nonexistent_base(self) -> None:
        """Should fail if base profile does not exist."""
        cfg = {"sandbox": {"profiles": {}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_create("new", from_profile="nonexistent")


class TestDeleteCommand:
    """Test sandbox delete command."""

    def test_delete_profile(self, capsys: pytest.CaptureFixture) -> None:
        """Should delete profile."""
        cfg = {
            "sandbox": {
                "active_profile": None,
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 0,
                    },
                    "minimal": {
                        "path": "arapuca",
                        "memory_mb": 1024,
                        "cpus": 1,
                        "pids": 128,
                        "timeout": 0,
                    },
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_delete("gpu")

        mock_save.assert_called_once()
        saved_cfg = mock_save.call_args[0][1]
        assert "gpu" not in saved_cfg["sandbox"]["profiles"]
        assert "minimal" in saved_cfg["sandbox"]["profiles"]

        captured = capsys.readouterr()
        assert "Deleted sandbox profile 'gpu'" in captured.out

    def test_delete_active_profile(self, capsys: pytest.CaptureFixture) -> None:
        """Should clear active_profile when deleting active."""
        cfg = {
            "sandbox": {
                "active_profile": "gpu",
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_delete("gpu")

        saved_cfg = mock_save.call_args[0][1]
        assert saved_cfg["sandbox"]["active_profile"] is None

    def test_delete_nonexistent(self) -> None:
        """Should fail to delete non-existent profile."""
        cfg = {"sandbox": {"profiles": {}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_delete("nonexistent")


class TestSetDefaultCommand:
    """Test sandbox set-default command."""

    def test_set_default(self, capsys: pytest.CaptureFixture) -> None:
        """Should promote profile to default settings."""
        cfg = {
            "sandbox": {
                "path": "arapuca",
                "memory_mb": 2048,
                "cpus": 2,
                "pids": 256,
                "timeout": 0,
                "active_profile": None,
                "profiles": {
                    "production": {
                        "path": "/usr/local/bin/arapuca",
                        "memory_mb": 16384,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_set_default("production")

        mock_save.assert_called_once()
        saved_cfg = mock_save.call_args[0][1]
        # Should copy all values to root
        assert saved_cfg["sandbox"]["path"] == "/usr/local/bin/arapuca"
        assert saved_cfg["sandbox"]["memory_mb"] == 16384
        assert saved_cfg["sandbox"]["cpus"] == 8
        assert saved_cfg["sandbox"]["pids"] == 512

        captured = capsys.readouterr()
        assert "Set 'production' as default sandbox settings" in captured.out

    def test_set_default_nonexistent(self) -> None:
        """Should fail if profile does not exist."""
        cfg = {"sandbox": {"profiles": {}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_set_default("nonexistent")


class TestActiveCommand:
    """Test sandbox active command."""

    def test_set_active_profile(self, capsys: pytest.CaptureFixture) -> None:
        """Should set active profile."""
        cfg = {
            "sandbox": {
                "active_profile": None,
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_active("gpu")

        saved_cfg = mock_save.call_args[0][1]
        assert saved_cfg["sandbox"]["active_profile"] == "gpu"

        captured = capsys.readouterr()
        assert "Set active sandbox profile to 'gpu'" in captured.out

    def test_clear_active_profile_with_empty_string(self, capsys: pytest.CaptureFixture) -> None:
        """Should clear active profile with empty string."""
        cfg = {
            "sandbox": {
                "active_profile": "gpu",
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_active("")

        saved_cfg = mock_save.call_args[0][1]
        assert saved_cfg["sandbox"]["active_profile"] is None

        captured = capsys.readouterr()
        assert "Cleared active sandbox profile" in captured.out

    def test_clear_active_profile_with_none(self, capsys: pytest.CaptureFixture) -> None:
        """Should clear active profile with 'none'."""
        cfg = {
            "sandbox": {
                "active_profile": "gpu",
                "profiles": {
                    "gpu": {
                        "path": "arapuca",
                        "memory_mb": 8192,
                        "cpus": 8,
                        "pids": 512,
                        "timeout": 0,
                    },
                },
            }
        }

        with (
            patch("arkai.config.load_config", return_value=cfg),
            patch("arkai.utils.save_yaml") as mock_save,
            patch("arkai.utils.get_config_home", return_value="/tmp"),
        ):
            sandbox.cmd_sandbox_active("none")

        saved_cfg = mock_save.call_args[0][1]
        assert saved_cfg["sandbox"]["active_profile"] is None

    def test_active_nonexistent(self) -> None:
        """Should fail if profile does not exist."""
        cfg = {"sandbox": {"profiles": {}}}

        with patch("arkai.config.load_config", return_value=cfg):
            with pytest.raises(SystemExit):
                sandbox.cmd_sandbox_active("nonexistent")


class TestDeduplicateVolumes:
    """Test volume deduplication logic."""

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert sandbox._deduplicate_volumes([]) == []

    def test_single_volume(self) -> None:
        """Single volume passes through unchanged."""
        assert sandbox._deduplicate_volumes(["/data"]) == ["/data"]

    def test_exact_duplicates_removed(self) -> None:
        """Exact duplicate volumes (path + flag) are deduplicated."""
        result = sandbox._deduplicate_volumes(["/data", "/data"])
        assert result == ["/data"]

    def test_exact_duplicates_with_flag(self) -> None:
        """Exact duplicates including flag are deduplicated."""
        result = sandbox._deduplicate_volumes(["/data:ro", "/data:ro"])
        assert result == ["/data:ro"]

    def test_different_paths_preserved(self) -> None:
        """Different paths are both preserved."""
        result = sandbox._deduplicate_volumes(["/data", "/logs"])
        assert result == ["/data", "/logs"]

    def test_conflict_same_path_different_flags_raises(self) -> None:
        """Same path with different flags raises RuntimeError."""
        with pytest.raises(RuntimeError, match="conflicting flags"):
            sandbox._deduplicate_volumes(["/data", "/data:ro"])

    def test_conflict_path_flagged_vs_plain(self) -> None:
        """Path with flag vs path without flag (different flag) raises RuntimeError."""
        with pytest.raises(RuntimeError):
            sandbox._deduplicate_volumes(["/data:ro", "/data"])

    def test_multiple_unique_volumes(self) -> None:
        """Multiple unique volumes all preserved."""
        vols = ["/data:ro", "/logs", "/tmp"]
        result = sandbox._deduplicate_volumes(vols)
        assert result == vols


class TestMergeVolumes:
    """Test volume merging in agent module."""

    def test_empty_inputs(self) -> None:
        """Both empty returns empty list."""
        assert agent_module._merge_volumes([], None) == []

    def test_profile_only(self) -> None:
        """Profile volumes returned when no CLI volumes."""
        result = agent_module._merge_volumes(["/data:ro"], None)
        assert result == ["/data:ro"]

    def test_cli_appends_to_profile(self) -> None:
        """CLI volumes are appended to profile volumes."""
        result = agent_module._merge_volumes(["/data:ro"], ["/logs"])
        assert "/data:ro" in result
        assert "/logs" in result

    def test_duplicate_volumes_deduplicated(self) -> None:
        """Duplicates across profile and CLI are removed."""
        result = agent_module._merge_volumes(["/data"], ["/data"])
        assert result.count("/data") == 1

    def test_conflict_raises(self) -> None:
        """Same path with conflicting flags raises RuntimeError."""
        with pytest.raises(RuntimeError):
            agent_module._merge_volumes(["/data:ro"], ["/data"])


class TestMergeEnvironment:
    """Test environment variable merging in agent module."""

    def test_empty_inputs(self) -> None:
        """Both empty returns empty dict."""
        assert agent_module._merge_environment({}, None) == {}

    def test_profile_only(self) -> None:
        """Profile env returned when no CLI env."""
        result = agent_module._merge_environment({"FOO": "bar"}, None)
        assert result == {"FOO": "bar"}

    def test_cli_adds_to_profile(self) -> None:
        """CLI env vars are added to profile env."""
        result = agent_module._merge_environment({"FOO": "bar"}, {"BAZ": "qux"})
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_cli_overrides_profile(self) -> None:
        """CLI env vars override profile env vars for same key."""
        result = agent_module._merge_environment({"FOO": "profile"}, {"FOO": "cli"})
        assert result["FOO"] == "cli"

    def test_none_profile(self) -> None:
        """None profile env treated as empty."""
        result = agent_module._merge_environment(None, {"FOO": "bar"})
        assert result == {"FOO": "bar"}
