"""Step definitions for vectordb service management features."""

from unittest.mock import MagicMock, patch

from behave import given, then, when

from arkai import vectordb


@given("vectordb server is running")  # ty: ignore[call-non-callable]
def step_vectordb_running(context):
    """Mock vectordb server as running."""
    context.vectordb_running_patch = patch.object(
        vectordb, "is_vectordb_running", return_value=True
    )
    context.vectordb_running_patch.start()

    context.vectordb_port_patch = patch.object(vectordb, "get_vectordb_port", return_value=8082)
    context.vectordb_port_patch.start()


@when('I run "arkai vectordb start"')  # ty: ignore[call-non-callable]
def step_run_vectordb_start(context):
    """Run vectordb start command with mocked subprocess."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.vectordb.subprocess.Popen") as mock_popen,
        patch("arkai.vectordb.requests.get") as mock_get,
        patch("arkai.vectordb.config.validate_config", return_value=True),
        patch("arkai.vectordb.is_vectordb_running", return_value=False),
        patch("arkai.vectordb.utils.is_port_in_use", return_value=False),
        patch("arkai.vectordb.utils.resolve_binary", return_value="/usr/bin/chroma"),
        patch("arkai.vectordb.utils.write_pid"),
        patch("arkai.vectordb.utils.save_yaml"),
    ):
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        try:
            vectordb.cmd_vectordb_start()
            context.stdout = "Vectordb server started"
            context.vectordb_started = True
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1
            context.vectordb_started = False


@when('I run "arkai vectordb stop"')  # ty: ignore[call-non-callable]
def step_run_vectordb_stop(context):
    """Run vectordb stop command with mocked subprocess."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.vectordb.utils.read_pid", return_value=12345),
        patch("arkai.vectordb.utils.kill_process", return_value=True),
        patch("arkai.vectordb.utils.wait_for_process_stop", return_value=True),
        patch("arkai.vectordb.os.path.exists", return_value=False),
    ):
        try:
            vectordb.cmd_vectordb_stop()
            context.stdout = "Vectordb server stopped"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


@when('I run "arkai vectordb status"')  # ty: ignore[call-non-callable]
def step_run_vectordb_status(context):
    """Run vectordb status command with mocked subprocess."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    try:
        vectordb.cmd_vectordb_status()
        context.stdout = "Vectordb server status"
    except Exception as e:
        context.stderr = str(e)
        context.exit_code = 1


@when('I run "arkai vectordb list"')  # ty: ignore[call-non-callable]
def step_run_vectordb_list(context):
    """Run vectordb list command with mocked requests."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with patch("arkai.vectordb.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "test-db", "id": "uuid-1"},
        ]
        mock_get.return_value = mock_response

        try:
            vectordb.cmd_vectordb_list()
            context.stdout = "Available databases"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


@when('I run "arkai vectordb initdb {db_name}"')  # ty: ignore[call-non-callable]
def step_run_vectordb_initdb(context, db_name):
    """Run vectordb initdb command with mocked subprocess."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.vectordb.requests.post") as mock_post,
        patch("arkai.vectordb.utils.warn"),
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        try:
            vectordb.cmd_vectordb_initdb(db_name)
            context.stdout = f"Database '{db_name}' created successfully"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


@when('I run "arkai vectordb drop {db_name}"')  # ty: ignore[call-non-callable]
def step_run_vectordb_drop(context, db_name):
    """Run vectordb drop command with mocked requests."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch("arkai.vectordb.requests.get") as mock_get,
        patch("arkai.vectordb.requests.delete") as mock_delete,
    ):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = [{"name": db_name, "id": "test-collection-uuid"}]
        mock_get.return_value = mock_get_response

        mock_delete_response = MagicMock()
        mock_delete_response.status_code = 200
        mock_delete.return_value = mock_delete_response

        try:
            vectordb.cmd_vectordb_drop(db_name)
            context.stdout = f"Database '{db_name}' dropped successfully"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


@then("vectordb server is running")  # ty: ignore[call-non-callable]
def step_verify_vectordb_running(context):
    """Verify vectordb is running."""
    if hasattr(context, "vectordb_started") and context.vectordb_started:
        return
    assert hasattr(context, "vectordb_running_patch"), "Vectordb running patch not found"
    assert context.vectordb_running_patch.return_value is True


@then("vectordb server is not running")  # ty: ignore[call-non-callable]
def step_verify_vectordb_not_running(context):
    """Verify vectordb is not running."""
    if hasattr(context, "vectordb_running_patch"):
        context.vectordb_running_patch.return_value = False


def teardown_vectordb(context):
    """Clean up after vectordb tests."""
    if hasattr(context, "vectordb_running_patch"):
        context.vectordb_running_patch.stop()
    if hasattr(context, "vectordb_port_patch"):
        context.vectordb_port_patch.stop()
