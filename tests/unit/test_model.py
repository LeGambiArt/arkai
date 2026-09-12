"""Unit tests for model management."""

import os
from unittest.mock import patch

import pytest

from arkai import model, providers, utils


class TestGetModelsDir:
    """Test model directory helpers."""

    def test_get_models_dir_returns_valid_path(self, tmp_path, monkeypatch):
        """Test that get_models_dir returns the correct path."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        assert model.get_models_dir() == os.path.join(str(tmp_path), "models")


class TestCmdModelList:
    """Test cmd_model_list function."""

    def test_list_no_models(self, tmp_path, monkeypatch, capsys):
        """Test listing when no local or cloned models exist."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))

        model.cmd_model_list()

        assert "No models found" in capsys.readouterr().out

    def test_list_local_and_git_models(self, tmp_path, monkeypatch, capsys):
        """Test listing local GGUF files and Git-cloned repositories."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "local.gguf").write_bytes(b"x")
        repository = models_dir / "providers" / "huggingface" / "org" / "model"
        (repository / ".git").mkdir(parents=True)
        (repository / "weights.gguf").write_bytes(b"x" * 1024)

        model.cmd_model_list()

        output = capsys.readouterr().out
        assert "Local GGUF models:" in output
        assert "local.gguf" in output
        assert "Provider models:" in output
        assert "hf:org/model" in output


class TestProviderCacheDiscovery:
    """Test provider cache discovery."""

    def test_get_git_models(self, tmp_path, monkeypatch):
        """Test listing cloned repositories and their file sizes."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        repository = tmp_path / "models" / "providers" / "huggingface" / "org" / "model"
        (repository / ".git").mkdir(parents=True)
        (repository / "weights.gguf").write_bytes(b"x" * 1024)
        (repository / "metadata").write_bytes(b"x" * 2048)

        assert providers.PROVIDERS["hf"].list_models() == [("org/model", "3.0K")]

    def test_get_models_without_cache(self, tmp_path, monkeypatch):
        """Test listing when the Git cache does not exist."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        assert providers.PROVIDERS["hf"].list_models() == []


class TestCmdModelDownload:
    """Test provider-backed model downloads."""

    def test_download_successful(self, tmp_path, monkeypatch, capsys):
        """Test successful model clone."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(providers.shutil, "which", lambda command: "/usr/bin/git")
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "", "")

            model.cmd_model_download("hf:test-org/test-model")

            mock_run.assert_called_once_with(
                [
                    "/usr/bin/git",
                    "clone",
                    "--depth",
                    "1",
                    "https://huggingface.co/test-org/test-model.git",
                    str(
                        tmp_path
                        / "models"
                        / "providers"
                        / "huggingface"
                        / "test-org"
                        / "test-model"
                    ),
                ],
                capture=False,
                timeout=None,
                env=mock_run.call_args.kwargs["env"],
            )
            assert mock_run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
            output = capsys.readouterr().out
            assert "Cloning hf:test-org/test-model" in output
            assert "Git transfer complete: hf:test-org/test-model" in output

    def test_download_existing_repository(self, tmp_path, monkeypatch):
        """Test updating an already cloned repository."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(providers.shutil, "which", lambda command: "/usr/bin/git")
        repository = tmp_path / "models" / "providers" / "huggingface" / "test-org" / "test-model"
        (repository / ".git").mkdir(parents=True)
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "", "")

            model.cmd_model_download("hf:test-org/test-model")

            assert mock_run.call_args[0][0] == [
                "/usr/bin/git",
                "-C",
                str(repository),
                "pull",
                "--ff-only",
            ]
            assert mock_run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"

    def test_download_accepts_huggingface_model_prefix(self, tmp_path, monkeypatch):
        """Test that the inference-style hf prefix is removed from downloads."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(providers.shutil, "which", lambda command: "/usr/bin/git")
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "", "")

            model.cmd_model_download("hf:test-org/test-model")

            command = mock_run.call_args[0][0]
            assert "https://huggingface.co/test-org/test-model.git" in command
            assert (
                str(tmp_path / "models" / "providers" / "huggingface" / "test-org" / "test-model")
                in command
            )

    def test_download_command_fails(self, tmp_path, monkeypatch):
        """Test when Git clone fails."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(providers.shutil, "which", lambda command: "/usr/bin/git")
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (1, "", "Repository not found")

            with pytest.raises(RuntimeError, match="Failed to download"):
                model.cmd_model_download("hf:test-org/test-model")

    def test_download_command_failure_without_stderr(self, tmp_path, monkeypatch):
        """Test Git failures without captured stderr produce a useful error."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(providers.shutil, "which", lambda command: "/usr/bin/git")
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (1, None, None)

            with pytest.raises(RuntimeError, match="non-zero exit status"):
                model.cmd_model_download("hf:test-org/test-model")

    def test_download_command_not_found(self, monkeypatch):
        """Test when Git is not available."""
        monkeypatch.setattr(providers.shutil, "which", lambda command: None)

        with pytest.raises(RuntimeError, match="git command not found"):
            model.cmd_model_download("hf:test-org/test-model")

    def test_download_rejects_invalid_repository(self):
        """Test that repository IDs cannot escape the cache directory."""
        with pytest.raises(ValueError, match="invalid path"):
            model.cmd_model_download("hf:../outside")


class TestCmdModelConvert:
    """Test model conversion command delegation."""

    def test_convert_successful(self):
        """Test successful model conversion."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "/path/to/output.gguf\n", "")

            model.cmd_model_convert("test-model", quantization="Q6_K")

            call_args = mock_run.call_args[0][0]
            assert call_args[0].endswith("arkai-convert")
            assert "test-model" in call_args
            assert "Q6_K" in call_args

    def test_convert_with_output_path(self):
        """Test conversion with explicit output path."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "/custom/output.gguf\n", "")

            model.cmd_model_convert("test-model", output="/custom/output.gguf")

            call_args = mock_run.call_args[0][0]
            assert "-o" in call_args
            assert "/custom/output.gguf" in call_args

    def test_convert_command_fails(self):
        """Test when conversion fails."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (1, "", "Conversion error")

            with pytest.raises(RuntimeError, match="Conversion failed"):
                model.cmd_model_convert("test-model")

    def test_convert_script_not_found(self):
        """Test when the conversion script cannot be started."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.side_effect = RuntimeError("Script not found")

            with pytest.raises(RuntimeError, match="Conversion script not found"):
                model.cmd_model_convert("test-model")

    def test_convert_rejects_ollama_model(self):
        """Test that already-GGUF Ollama artifacts are not reconverted."""
        with pytest.raises(RuntimeError, match="already downloaded as GGUF"):
            model.cmd_model_convert("ollama:llama3")


class TestCmdModelRemove:
    """Test local model removal."""

    def test_remove_model(self, tmp_path, monkeypatch):
        """Test removing a local GGUF file."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.gguf").write_bytes(b"model")

        model.cmd_model_remove("model.gguf")

        assert not (models_dir / "model.gguf").exists()

    def test_remove_missing_model(self, tmp_path, monkeypatch):
        """Test removing a missing model produces a clear error."""
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))

        with pytest.raises(RuntimeError, match="Model not found"):
            model.cmd_model_remove("missing.gguf")

    def test_remove_huggingface_model(self, capsys):
        """Test removing a provider-cached model by its qualified reference."""
        with patch.object(model.providers, "remove_model") as remove_model:
            model.cmd_model_remove("hf:mlx-community/Qwen3-4B-4bit")

        remove_model.assert_called_once_with("hf:mlx-community/Qwen3-4B-4bit")
        assert "Removed hf:mlx-community/Qwen3-4B-4bit" in capsys.readouterr().out

    def test_remove_unknown_provider_model_reports_error(self):
        """Test provider removal preserves a clear missing-model error."""
        with patch.object(
            model.providers,
            "remove_model",
            side_effect=RuntimeError("Model not found: hf:org/missing"),
        ):
            with pytest.raises(RuntimeError, match="Model not found: hf:org/missing"):
                model.cmd_model_remove("hf:org/missing")
