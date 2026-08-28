import subprocess

import pytest

from arkai import utils


class TestPathResolution:
    def test_config_home_uses_xdg_config_home(self, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/config")
        assert utils.get_config_home() == "/custom/config/arkai"

    def test_config_home_fallback_macos(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", "/Users/test")
        assert utils.get_config_home() == "/Users/test/.config/arkai"

    def test_data_home_always_local_share(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")
        monkeypatch.setenv("HOME", "/Users/test")
        assert utils.get_data_home() == "/Users/test/.local/share/arkai"

    def test_data_home_no_xdg_override(self, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setenv("HOME", "/Users/test")
        assert utils.get_data_home() == "/Users/test/.local/share/arkai"

    def test_pid_dir_uses_local_state(self, monkeypatch):
        monkeypatch.setenv("HOME", "/Users/test")
        assert utils.get_pid_dir() == "/Users/test/.local/state/arkai"


class TestGPUDetection:
    def test_detect_gpu_metal_on_macos(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("arkai.utils.run_command", lambda cmd, **kw: (0, "1", ""))
        result = utils.detect_gpu()
        assert result == "metal"

    def test_detect_gpu_cuda_on_linux(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr(
            "shutil.which", lambda x: "/usr/bin/nvidia-smi" if x == "nvidia-smi" else None
        )
        monkeypatch.setattr("arkai.utils.run_command", lambda cmd, **kw: (0, "", ""))
        result = utils.detect_gpu()
        assert result == "cuda"

    def test_detect_gpu_fallback_cpu(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("shutil.which", lambda x: None)
        result = utils.detect_gpu()
        assert result == "cpu"


class TestYAMLHandling:
    def test_load_yaml_valid(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("agent:\n  name: opencode\n")
        result = utils.load_yaml(str(yaml_file))
        assert result["agent"]["name"] == "opencode"

    def test_load_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            utils.load_yaml("/nonexistent/file.yaml")

    def test_save_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        data = {"agent": {"name": "crush"}}
        utils.save_yaml(str(yaml_file), data)
        result = utils.load_yaml(str(yaml_file))
        assert result["agent"]["name"] == "crush"

    def test_merge_configs(self):
        base = {"agent": {"name": "opencode"}, "inference": {"port": 8081}}
        override = {"agent": {"name": "crush"}}
        result = utils.merge_configs(base, override)
        assert result["agent"]["name"] == "crush"
        assert result["inference"]["port"] == 8081


class TestPIDManagement:
    def test_write_and_read_pid(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        utils.write_pid(str(pid_file), 12345)
        assert utils.read_pid(str(pid_file)) == 12345

    def test_read_pid_nonexistent(self, tmp_path):
        pid_file = tmp_path / "nonexistent.pid"
        assert utils.read_pid(str(pid_file)) is None

    def test_read_pid_invalid_content(self, tmp_path):
        pid_file = tmp_path / "invalid.pid"
        pid_file.write_text("not_a_number")
        assert utils.read_pid(str(pid_file)) is None

    def test_kill_process_success(self, monkeypatch):
        # Mock os.kill to not actually kill anything
        monkeypatch.setattr("os.kill", lambda pid, sig: None)
        result = utils.kill_process(9999)
        assert result is True

    def test_kill_process_failure(self, monkeypatch):
        # Mock os.kill to raise ProcessLookupError
        monkeypatch.setattr("os.kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
        result = utils.kill_process(9999)
        assert result is False


class TestErrorHandling:
    def test_error_exits(self, capsys):
        utils.error("Test error", 2)
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err

    def test_warn_to_stderr(self, capsys):
        utils.warn("Test warning")
        captured = capsys.readouterr()
        assert "Warning: Test warning" in captured.err

    def test_info_to_stdout(self, capsys):
        utils.info("Test info")
        captured = capsys.readouterr()
        assert "Test info" in captured.out


class TestBinaryResolution:
    def test_resolve_binary_absolute_path(self, tmp_path):
        # Create a fake executable
        fake_bin = tmp_path / "fake"
        fake_bin.write_text("#!/bin/sh\necho test\n")
        fake_bin.chmod(0o755)
        result = utils.resolve_binary(str(fake_bin))
        assert result == str(fake_bin)

    def test_resolve_binary_tilde_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_bin = bin_dir / "test"
        fake_bin.write_text("#!/bin/sh\necho test\n")
        fake_bin.chmod(0o755)
        result = utils.resolve_binary("~/bin/test")
        assert result == str(fake_bin)

    def test_resolve_binary_in_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/python" if x == "python" else None)
        result = utils.resolve_binary("python")
        assert result == "/usr/bin/python"

    def test_resolve_binary_not_found(self):
        with pytest.raises(RuntimeError):
            utils.resolve_binary("/nonexistent/binary")

    def test_resolve_binary_not_in_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda x: None)
        with pytest.raises(RuntimeError):
            utils.resolve_binary("nonexistent-binary")


class TestProcessExecution:
    def test_run_command_success(self):
        code, stdout, stderr = utils.run_command(["echo", "hello"])
        assert code == 0
        assert "hello" in stdout

    def test_run_command_failure(self):
        code, _, _ = utils.run_command(["false"])
        assert code != 0

    def test_run_command_without_capture(self):
        code, stdout, stderr = utils.run_command(["echo", "test"], capture=False)
        assert code == 0
        assert stdout is None

    def test_run_command_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired("cmd", 1)),
        )
        with pytest.raises(RuntimeError):
            utils.run_command(["sleep", "10"])

    def test_run_command_not_found(self, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError())
        )
        with pytest.raises(RuntimeError):
            utils.run_command(["nonexistent-command"])


class TestYAMLErrors:
    def test_load_yaml_invalid_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("invalid: yaml: syntax:")
        with pytest.raises(RuntimeError):
            utils.load_yaml(str(yaml_file))

    def test_load_yaml_empty_file(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        result = utils.load_yaml(str(yaml_file))
        assert result == {}

    def test_save_yaml_creates_parent_dirs(self, tmp_path):
        yaml_file = tmp_path / "subdir" / "config.yaml"
        data = {"test": "value"}
        utils.save_yaml(str(yaml_file), data)
        assert yaml_file.exists()
        result = utils.load_yaml(str(yaml_file))
        assert result["test"] == "value"


class TestConfigHomeErrors:
    def test_config_home_no_home_env(self, monkeypatch):
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        with pytest.raises(RuntimeError):
            utils.get_config_home()

    def test_data_home_no_home_env(self, monkeypatch):
        monkeypatch.delenv("HOME", raising=False)
        with pytest.raises(RuntimeError):
            utils.get_data_home()

    def test_pid_dir_no_home_env(self, monkeypatch):
        monkeypatch.delenv("HOME", raising=False)
        with pytest.raises(RuntimeError):
            utils.get_pid_dir()


class TestMessageLevel:
    """Test message level functionality."""

    def test_message_level_default(self):
        """Test that default message level is INFO."""
        assert utils.get_message_level() == utils.MessageLevel.INFO

    def test_set_message_level(self):
        """Test setting message level."""
        original = utils.get_message_level()
        try:
            utils.set_message_level(utils.MessageLevel.ERROR)
            assert utils.get_message_level() == utils.MessageLevel.ERROR

            utils.set_message_level(utils.MessageLevel.WARN)
            assert utils.get_message_level() == utils.MessageLevel.WARN

            utils.set_message_level(utils.MessageLevel.INFO)
            assert utils.get_message_level() == utils.MessageLevel.INFO
        finally:
            utils.set_message_level(original)

    def test_message_level_values(self):
        """Test message level ordering."""
        assert utils.MessageLevel.ERROR.value < utils.MessageLevel.WARN.value
        assert utils.MessageLevel.WARN.value < utils.MessageLevel.INFO.value
