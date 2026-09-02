"""Step definitions for server status monitoring features."""

import sys
from contextlib import ExitStack
from io import StringIO
from unittest.mock import patch

from behave import when

from arkai import engine as engine_module
from arkai import status as status_module
from arkai import utils
from arkai import vectordb as vectordb_module
from arkai import wtmcp as wtmcp_module


@when('I run "arkai status"')  # ty: ignore[call-non-callable]
def step_run_status(context):
    """Run arkai status command and capture output."""
    old_stdout = sys.stdout
    stdout_capture = StringIO()
    sys.stdout = stdout_capture

    context.exit_code = 0
    try:
        # Check if servers are mocked to be running via Given steps
        inference_running = hasattr(context, "is_running_patch") and context.is_running_patch
        vectordb_running = (
            hasattr(context, "vectordb_running_patch") and context.vectordb_running_patch
        )

        with ExitStack() as stack:
            # Apply patches for servers not explicitly mocked as running
            if not inference_running:
                stack.enter_context(
                    patch.object(engine_module, "is_inference_running", return_value=False)
                )
            if not vectordb_running:
                stack.enter_context(
                    patch.object(vectordb_module, "is_vectordb_running", return_value=False)
                )
            # Always mock wtmcp to be not running
            stack.enter_context(
                patch.object(wtmcp_module, "_get_running_instances", return_value=[])
            )

            # If vectordb is mocked as running, also mock the state file loading
            if vectordb_running:
                stack.enter_context(
                    patch.object(
                        utils,
                        "read_pid",
                        return_value=20050,
                    )
                )
                stack.enter_context(
                    patch.object(
                        utils,
                        "load_yaml",
                        return_value={"port": 8082, "databases": []},
                    )
                )

            status_module.cmd_status()
    except Exception:
        context.exit_code = 1
    finally:
        sys.stdout = old_stdout
        context.stdout = stdout_capture.getvalue()
