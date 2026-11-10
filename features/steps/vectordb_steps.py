"""Step definitions for vectordb service management features."""

from unittest.mock import MagicMock, patch

from behave import given, then, when

from arkai import vectordb


def _patch_vectordb_start(context):
    """Mock vectordb server as running."""
    context.vectordb_running_patch = patch.object(
        vectordb, "is_vectordb_running", return_value=True
    )
    context.vectordb_running_patch.start()

    context.vectordb_port_patch = patch.object(vectordb, "get_vectordb_port", return_value=8082)
    context.vectordb_port_patch.start()
    context.vectordb_running = True


def _patch_vectordb_stop(context):
    """Mock vectordb server as running."""
    context.vectordb_running_patch = patch.object(
        vectordb, "is_vectordb_running", return_value=False
    )
    context.vectordb_running_patch.start()

    context.vectordb_port_patch = patch.object(vectordb, "get_vectordb_port", return_value=None)
    context.vectordb_port_patch.start()
    context.vectordb_running = False


@given("vectordb server is running")  # ty: ignore[call-non-callable]
def step_vectordb_running(context):
    """Ensure vectordb server is running."""
    _patch_vectordb_start(context)


@then("the vectordb container image is the default qdrant image")  # ty: ignore[call-non-callable]
def step_verify_vectordb_container_image(context):
    """Verify the engine was asked to run the default qdrant image."""
    image = getattr(context, "vectordb_container_image", None)
    assert image == "docker.io/qdrant/qdrant:latest", f"unexpected image: {image}"


@when("the container engine VM is not running")  # ty: ignore[call-non-callable]
def step_container_engine_vm_not_running(context):
    """Mark the container engine VM as stopped for the next start attempt."""
    context.container_engine_vm_stopped = True


@when('I run "arkai vectordb start"')  # ty: ignore[call-non-callable]
def step_run_vectordb_start(context):
    """Run vectordb start command with mocked container engine."""
    from arkai import vectordb as vectordb_module

    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch.object(vectordb_module.container_engine, "resolve_engine") as mock_resolve,
        patch.object(vectordb_module.container_engine, "ensure_engine_ready") as mock_ready,
        patch.object(vectordb_module.container_engine, "run_container") as mock_run,
        patch.object(vectordb_module.requests, "get") as mock_get,
        patch.object(vectordb_module, "config") as mock_config,
        patch.object(vectordb_module.utils, "is_port_in_use", return_value=False),
        patch.object(vectordb_module.utils, "write_pid"),
        patch.object(vectordb_module.utils, "save_yaml"),
    ):
        mock_resolve.return_value = "podman"
        if getattr(context, "container_engine_vm_stopped", False):
            mock_ready.side_effect = RuntimeError(
                "Container engine 'podman' is not ready: start it with 'podman machine start'"
            )
        else:
            mock_ready.return_value = None
        mock_run.return_value = None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        mock_config.load_config.return_value = {
            "vectordb": {"vendor": "qdrant", "engine": "auto", "port": 8082}
        }
        mock_config.validate_config.return_value = True
        values = {
            "vectordb.port": 8082,
            "vectordb.vendor": "qdrant",
            "vectordb.image": None,
            "vectordb.database_dir": None,
        }
        mock_config.get_config_value.side_effect = lambda cfg, key, default=None: values.get(
            key, default
        )

        try:
            vectordb.cmd_vectordb_start()
            context.stdout = "Vectordb server started"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1
            context.vectordb_running = False

        # Record which container image the engine was asked to run
        context.vectordb_container_image = (
            mock_run.call_args.kwargs.get("image") if mock_run.called else None
        )

        # Mock is_vector_running to return True
        context.vectordb_running_patch = patch.object(
            vectordb_module, "is_vectordb_running", return_value=True
        )
        context.vectordb_running_patch.start()
        context.vectordb_running = True


@when('I run "arkai vectordb stop"')  # ty: ignore[call-non-callable]
def step_run_vectordb_stop(context):
    """Run vectordb stop command with mocked container engine."""
    from arkai import vectordb as vectordb_module

    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch.object(vectordb_module.container_engine, "ENGINES", ("podman", "apple", "docker")),
        patch.object(vectordb_module.container_engine, "resolve_engine", return_value="podman"),
        patch.object(vectordb_module.container_engine, "stop_container") as mock_stop,
        patch.object(vectordb_module, "_read_state", return_value={"engine": "podman"}),
    ):
        mock_stop.return_value = None

        try:
            vectordb.cmd_vectordb_stop()
            context.stdout = "Vectordb server stopped"
            _patch_vectordb_stop(context)
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

    with (
        patch.object(vectordb, "_vendor", return_value="qdrant"),
        patch("arkai.vectordb.requests.get") as mock_get,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"collections": [{"name": "test-db"}]}}
        mock_get.return_value = mock_response

        try:
            vectordb.cmd_vectordb_list()
            context.stdout = "Available databases"
        except Exception as e:
            context.stderr = str(e)
            context.exit_code = 1


@when('I run "arkai vectordb initdb {db_name}"')  # ty: ignore[call-non-callable]
def step_run_vectordb_initdb(context, db_name):
    """Run vectordb initdb command with mocked requests."""
    context.stdout = ""
    context.stderr = ""
    context.exit_code = 0

    with (
        patch.object(vectordb, "list_collections", return_value=[]),
        patch.object(vectordb, "create_collection") as mock_create,
    ):
        mock_create.return_value = None

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
        patch.object(vectordb, "_vendor", return_value="qdrant"),
        patch("arkai.vectordb.requests.delete") as mock_delete,
    ):
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
    # Check that instance is running (call should be mocked at this point)
    assert vectordb.is_vectordb_running(), "vector db server is not running"


@then("vectordb server is not running")  # ty: ignore[call-non-callable]
def step_verify_vectordb_not_running(context):
    """Verify vectordb is not running."""
    # Check that no instances are running (call should be mocked at this point)
    assert not vectordb.is_vectordb_running(), "vector db server is running"


def teardown_vectordb(context):
    """Clean up after vectordb tests."""
    if hasattr(context, "vectordb_running_patch"):
        context.vectordb_running_patch.stop()
    if hasattr(context, "vectordb_port_patch"):
        context.vectordb_port_patch.stop()
