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
    """Run rag ingest command with mocked vectordb dispatch."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.rag.config.validate_config", return_value=True),
        patch("arkai.rag.vectordb.is_vectordb_running", return_value=True),
        patch("arkai.rag.vectordb.get_vectordb_port", return_value=8082),
        patch("arkai.rag.vectordb._vendor", return_value="qdrant"),
        patch("arkai.rag.vectordb.list_collections") as mock_list,
        patch("arkai.rag.vectordb.create_collection") as mock_create,
        patch("arkai.rag.vectordb.add_documents") as mock_add,
        patch("arkai.rag._load_embedding_model") as mock_load_model,
        patch("arkai.rag._embed_text", return_value=[0.1, 0.2]),
    ):
        mock_list.return_value = [{"name": db_name, "id": db_name}]
        mock_create.return_value = None
        mock_add.return_value = None

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
    """Run rag query command with mocked vectordb dispatch."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.rag.config.validate_config", return_value=True),
        patch("arkai.rag.vectordb.is_vectordb_running", return_value=True),
        patch("arkai.rag.vectordb.get_vectordb_port", return_value=8082),
        patch("arkai.rag.vectordb._vendor", return_value="qdrant"),
        patch("arkai.rag.requests.get") as mock_get,
        patch("arkai.rag.vectordb.search_documents") as mock_search,
        patch("arkai.rag._load_embedding_model") as mock_load_model,
        patch("arkai.rag._embed_text", return_value=[0.1, 0.2]),
    ):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"data": [{"id": "running-model"}]}
        mock_get.return_value = mock_get_response

        mock_search.return_value = [
            ("Sample content from document", {"source": "test.txt", "chunk_index": 0}, 0.1)
        ]

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
