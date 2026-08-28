"""Unit tests for model management."""

import os
from unittest.mock import patch

import pytest

from arkai import model, utils


class TestGetModelsDir:
    """Test get_models_dir function."""

    def test_get_models_dir_returns_valid_path(self, tmp_path, monkeypatch):
        """Test that get_models_dir returns the correct path."""
        expected_path = os.path.join(str(tmp_path), "models")
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))

        result = model.get_models_dir()
        assert result == expected_path


class TestCmdModelList:
    """Test cmd_model_list function."""

    def test_list_no_models(self, tmp_path, monkeypatch, capsys):
        """Test listing when no models exist locally or in HF cache."""
        # Setup: no local models directory, no HF cached models
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(model, "_get_huggingface_cached_models", lambda: [])

        model.cmd_model_list()

        captured = capsys.readouterr()
        assert "No models found" in captured.out

    def test_list_only_local_gguf_models(self, tmp_path, monkeypatch, capsys):
        """Test listing when only local GGUF files exist."""
        # Setup: create local GGUF files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model1.gguf").write_bytes(b"x" * 1000000)  # 1 MB
        (models_dir / "model2.gguf").write_bytes(b"x" * 2000000)  # 2 MB

        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(model, "_get_huggingface_cached_models", lambda: [])

        model.cmd_model_list()

        captured = capsys.readouterr()
        assert "Local GGUF models:" in captured.out
        assert "model1.gguf" in captured.out
        assert "model2.gguf" in captured.out
        assert "HuggingFace cached models: none" in captured.out

    def test_list_only_huggingface_models(self, tmp_path, monkeypatch, capsys):
        """Test listing when only HuggingFace cached models exist."""
        # Setup: no local models, but HF cached models
        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        hf_models = [
            ("meta-llama/Llama-2-7b", "3.4G"),
            ("mistralai/Mistral-7B", "3.2G"),
        ]
        monkeypatch.setattr(model, "_get_huggingface_cached_models", lambda: hf_models)

        model.cmd_model_list()

        captured = capsys.readouterr()
        assert "Local GGUF models" in captured.out
        assert "HuggingFace cached models:" in captured.out
        assert "meta-llama/Llama-2-7b" in captured.out
        assert "mistralai/Mistral-7B" in captured.out

    def test_list_both_local_and_huggingface_models(self, tmp_path, monkeypatch, capsys):
        """Test listing when both local and HF cached models exist."""
        # Setup: create local GGUF files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "local.gguf").write_bytes(b"x" * 1000000)

        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        hf_models = [("meta-llama/Llama-2-7b", "3.4G")]
        monkeypatch.setattr(model, "_get_huggingface_cached_models", lambda: hf_models)

        model.cmd_model_list()

        captured = capsys.readouterr()
        assert "Local GGUF models:" in captured.out
        assert "local.gguf" in captured.out
        assert "HuggingFace cached models:" in captured.out
        assert "meta-llama/Llama-2-7b" in captured.out

    def test_list_local_models_sorted_order(self, tmp_path, monkeypatch, capsys):
        """Test that local GGUF models are displayed in sorted order."""
        # Setup: create local GGUF files in non-alphabetical order
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "zebra.gguf").write_bytes(b"x" * 1000)
        (models_dir / "apple.gguf").write_bytes(b"x" * 1000)
        (models_dir / "middle.gguf").write_bytes(b"x" * 1000)

        monkeypatch.setattr(utils, "get_data_home", lambda: str(tmp_path))
        monkeypatch.setattr(model, "_get_huggingface_cached_models", lambda: [])

        model.cmd_model_list()

        captured = capsys.readouterr()
        # Check that models appear in sorted order
        apple_pos = captured.out.find("apple.gguf")
        middle_pos = captured.out.find("middle.gguf")
        zebra_pos = captured.out.find("zebra.gguf")
        assert apple_pos < middle_pos < zebra_pos


class TestGetHuggingfaceCachedModels:
    """Test _get_huggingface_cached_models function."""

    def test_get_hf_models_successful(self):
        """Test successful retrieval of HF cached models."""
        hf_output = (
            "ID                                SIZE LAST_ACCESSED LAST_MODIFIED REFS\n"
            "model/meta-llama/Llama-2-7b       3.4G 10 hours ago  5 days ago    ['main']\n"
            "model/mistralai/Mistral-7B        3.2G 2 hours ago   3 days ago    ['main']\n"
            "Found 2 repo(s) for a total of 2 revision(s) and 6.6G on disk.\n"
        )

        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, hf_output, "")

            result = model._get_huggingface_cached_models()

            assert len(result) == 2
            assert result[0][0] == "meta-llama/Llama-2-7b"
            assert result[0][1] == "3.4G"
            assert result[1][0] == "mistralai/Mistral-7B"
            assert result[1][1] == "3.2G"
            mock_run.assert_called_once_with(["hf", "cache", "ls"])

    def test_get_hf_models_no_cache(self):
        """Test when hf cache is empty."""
        hf_output = (
            "ID SIZE LAST_ACCESSED LAST_MODIFIED REFS\n"
            "Found 0 repo(s) for a total of 0 revision(s) and 0B on disk.\n"
        )

        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, hf_output, "")

            result = model._get_huggingface_cached_models()

            assert result == []

    def test_get_hf_models_command_failed(self):
        """Test when hf cache command fails."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (1, "", "error message")

            result = model._get_huggingface_cached_models()

            assert result == []

    def test_get_hf_models_command_not_found(self):
        """Test when hf command is not available."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.side_effect = RuntimeError("hf not found")

            result = model._get_huggingface_cached_models()

            assert result == []

    def test_get_hf_models_with_warnings(self):
        """Test parsing output with warnings."""
        hf_output = (
            "ID                           SIZE LAST_ACCESSED LAST_MODIFIED REFS\n"
            "model/apple/DiffuCoder-7B-Base 15.2G 10 hours ago  22 hours ago  ['main']\n"
            "Found 1 repo(s) for a total of 1 revision(s) and 15.2G on disk.\n"
            "Warning: Found 5 cache inconsistencies. Re-run with `--show-warnings`.\n"
        )

        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, hf_output, "")

            result = model._get_huggingface_cached_models()

            assert len(result) == 1
            assert result[0][0] == "apple/DiffuCoder-7B-Base"
            assert result[0][1] == "15.2G"

    def test_get_hf_models_various_sizes(self):
        """Test parsing models with various size formats."""
        hf_output = (
            "ID              SIZE LAST_ACCESSED LAST_MODIFIED REFS\n"
            "model/small/model 500M 1 hour ago   1 day ago     ['main']\n"
            "model/medium/model 2.5G 2 hours ago  2 days ago    ['main']\n"
            "model/large/model 50G  3 hours ago  3 days ago    ['main']\n"
            "Found 3 repo(s).\n"
        )

        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, hf_output, "")

            result = model._get_huggingface_cached_models()

            assert len(result) == 3
            assert result[0] == ("small/model", "500M")
            assert result[1] == ("medium/model", "2.5G")
            assert result[2] == ("large/model", "50G")

    def test_get_hf_models_filters_invalid_entries(self):
        """Test that invalid entries without '/' are filtered out."""
        hf_output = (
            "ID         SIZE LAST_ACCESSED LAST_MODIFIED REFS\n"
            "model/valid/repo 1G   1 hour ago   1 day ago     ['main']\n"
            "invalid_entry    2G   2 hours ago  2 days ago    ['main']\n"
            "Found 2 repo(s).\n"
        )

        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, hf_output, "")

            result = model._get_huggingface_cached_models()

            # Should only include the valid entry with 'model/' prefix
            assert len(result) == 1
            assert result[0][0] == "valid/repo"

    def test_get_hf_models_long_repo_names(self):
        """Test that long repo names (>40 chars) are not truncated."""
        hf_output = (
            "ID SIZE LAST_ACCESSED LAST_MODIFIED REFS\n"
            "model/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF 53.9G 19 hours 19 hours ['main']\n"
            "Found 1 repo(s).\n"
        )

        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, hf_output, "")

            result = model._get_huggingface_cached_models()

            assert len(result) == 1
            # Verify full repo name is preserved
            assert result[0][0] == "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
            assert result[0][1] == "53.9G"


class TestCmdModelDownload:
    """Test cmd_model_download function."""

    def test_download_successful(self):
        """Test successful model download."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "", "")

            model.cmd_model_download("test-org/test-model")

            mock_run.assert_called_once_with(
                ["hf", "download", "test-org/test-model", "--repo-type", "model"],
                capture=False,
                timeout=None,
            )

    def test_download_command_fails(self):
        """Test when hf download command fails."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (1, "", "Download failed")

            with pytest.raises(RuntimeError, match="Failed to download"):
                model.cmd_model_download("test-org/test-model")

    def test_download_command_not_found(self):
        """Test when hf command is not available."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.side_effect = RuntimeError("Command not found: hf")

            with pytest.raises(RuntimeError, match="hf command not found"):
                model.cmd_model_download("test-org/test-model")


class TestCmdModelConvert:
    """Test cmd_model_convert function."""

    def test_convert_successful(self):
        """Test successful model conversion."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (0, "/path/to/output.gguf\n", "")

            model.cmd_model_convert("test-model", quantization="Q6_K")

            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert call_args[0].endswith("arkai-convert")
            assert "test-model" in call_args
            assert "-q" in call_args
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
        """Test when conversion command fails."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.return_value = (1, "", "Conversion error")

            with pytest.raises(RuntimeError, match="Conversion failed"):
                model.cmd_model_convert("test-model")

    def test_convert_script_not_found(self):
        """Test when conversion script is not found."""
        with patch.object(utils, "run_command") as mock_run:
            mock_run.side_effect = RuntimeError("Script not found")

            with pytest.raises(RuntimeError, match="Conversion script not found"):
                model.cmd_model_convert("test-model")
