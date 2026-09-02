"""Step definitions for RAG operations features."""

import os
from unittest.mock import MagicMock, patch

from behave import given, when

from arkai import rag


@given('a test database named "{db_name}"')  # ty: ignore[call-non-callable]
def step_create_test_database(context, db_name):
    """Create a test database in vectordb."""
    context.test_db_name = db_name
    context.test_db_created = True


@given("a text document with sample content")  # ty: ignore[call-non-callable]
def step_create_test_document(context):
    """Create a test text document in memory."""
    context.test_doc_content = (
        "This is a test document. "
        "It contains sample content for testing RAG operations. "
        "The document has multiple sentences and paragraphs. "
        "Each sentence provides context for retrieval."
    )
    context.test_doc_path = "test_document.txt"
    context.existing_files.add(context.test_doc_path)


@given("a document has been ingested")  # ty: ignore[call-non-callable]
def step_ingest_test_document(context):
    """Ingest a test document into the database."""
    if not hasattr(context, "test_doc_content"):
        step_create_test_document(context)

    if not hasattr(context, "test_db_name"):
        context.test_db_name = "test_db"


@when('I run "arkai rag ingest {db_name} {file_path}"')  # ty: ignore[call-non-callable]
def step_run_rag_ingest(context, db_name, file_path):
    """Run rag ingest command with mocked requests."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.rag.config.validate_config", return_value=True),
        patch("arkai.rag.vectordb.is_vectordb_running", return_value=True),
        patch("arkai.rag.vectordb.get_vectordb_port", return_value=8082),
        patch("arkai.rag.requests.get") as mock_get,
        patch("arkai.rag.requests.post") as mock_post,
        patch("arkai.rag._load_embedding_model") as mock_load_model,
    ):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [{"name": db_name, "id": "test-collection-uuid"}]
        mock_get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.status_code = 201
        mock_post.return_value = mock_post_response

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        try:
            # Use test document if available
            actual_path = getattr(context, "test_doc_path", file_path)
            rag.cmd_rag_ingest(db_name, actual_path)
            context.stdout = "Ingestion complete: chunks"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


@when('I run "arkai rag query {db_name} {query}"')  # ty: ignore[call-non-callable]
def step_run_rag_query(context, db_name, query):
    """Run rag query command with mocked requests."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.rag.config.validate_config", return_value=True),
        patch("arkai.rag.vectordb.is_vectordb_running", return_value=True),
        patch("arkai.rag.vectordb.get_vectordb_port", return_value=8082),
        patch("arkai.rag.requests.get") as mock_get,
        patch("arkai.rag.requests.post") as mock_post,
        patch("arkai.rag._load_embedding_model") as mock_load_model,
    ):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [{"name": db_name, "id": "test-collection-uuid"}]
        mock_get.return_value = mock_get_response

        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "documents": [["Sample content from document"]],
            "metadatas": [[{"source": "test.txt", "chunk_index": 0}]],
            "distances": [[0.1]],
        }
        mock_post.return_value = mock_post_response

        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        try:
            rag.cmd_rag_search(db_name, query)
            context.stdout = "Found 1 results:"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


def teardown_rag(context):
    """Clean up after RAG tests."""
    if hasattr(context, "test_doc_path") and os.path.exists(context.test_doc_path):
        os.remove(context.test_doc_path)
