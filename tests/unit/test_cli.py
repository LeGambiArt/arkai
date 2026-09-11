"""Tests for the top-level command-line interface."""

import sys
from unittest.mock import patch

import pytest

from arkai import cli, status, utils


def test_debug_enables_maximum_logging_and_prints_traceback(capsys) -> None:
    """Debug mode enables info output and prints the complete exception traceback."""
    with (
        patch.object(status, "exec_cmd", side_effect=RuntimeError("boom")),
        patch.object(utils, "set_message_level") as set_level,
        patch.object(sys, "argv", ["arkai", "--debug", "status"]),
    ):
        with pytest.raises(SystemExit, match="1"):
            cli.main()

    set_level.assert_called_once_with(utils.MessageLevel.INFO)
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_without_debug_prints_concise_exception(capsys) -> None:
    """Default failures do not include a traceback."""
    with (
        patch.object(status, "exec_cmd", side_effect=RuntimeError("boom")),
        patch.object(sys, "argv", ["arkai", "status"]),
    ):
        with pytest.raises(SystemExit, match="1"):
            cli.main()

    assert "Traceback (most recent call last)" not in capsys.readouterr().err
