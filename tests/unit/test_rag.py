"""Unit tests for RAG command dispatch."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from arkai import rag


@pytest.mark.parametrize("rag_cmd", ["ingest", "query"])
def test_rag_command_warns_about_incomplete_functionality(rag_cmd):
    """Test every RAG operation displays the incomplete functionality warning."""
    args = SimpleNamespace(
        rag_cmd=rag_cmd,
        db_name="test-db",
        file_path="document.txt",
        chunk_size=None,
        query="test query",
        results=None,
    )

    with patch("arkai.rag.utils") as mock_utils:
        with patch("arkai.rag.cmd_rag_ingest") as mock_ingest:
            with patch("arkai.rag.cmd_rag_search") as mock_search:
                rag.exec_cmd(args)

    mock_utils.warn.assert_called_once_with("RAG commands are not fully functional yet.")
    if rag_cmd == "ingest":
        mock_ingest.assert_called_once_with("test-db", "document.txt", None)
        mock_search.assert_not_called()
    else:
        mock_search.assert_called_once_with("test-db", "test query", None)
        mock_ingest.assert_not_called()
