"""Native model-store workflow steps without accessing a real Splash installation."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

from behave import given, then, when

from arkai import model, model_splash
from arkai.model_store import ModelEntry

REFERENCE = "hf:owner/splash-model"


@given("the Splash model installer bridge is available")  # ty: ignore[call-non-callable]
def splash_store_available(context: Any) -> None:
    """Provide synthetic installer responses, never launching a runtime or downloader."""

    def call(operation: str, **settings: Any) -> dict:
        context.native_operation = operation
        context.native_settings = settings
        return {
            "path": "/fixture/splash-model",
            "paths": [],
            "dry_run": settings.get("dry_run", False),
        }

    for replacement in (
        patch.object(model_splash.SplashModelStore, "call", side_effect=call),
        patch.object(
            model_splash.SplashModelStore,
            "list_models",
            return_value=[ModelEntry(REFERENCE, "fixture", Path("/fixture/splash-model"), True)],
        ),
        patch("arkai.inference_splash.SplashBackend.check_environment"),
    ):
        replacement.start()
        context.add_cleanup(replacement.stop)


@when("I download a model through the Splash store")  # ty: ignore[call-non-callable]
def native_download(context: Any) -> None:
    """Download via the native installer, not a provider Git clone."""
    model.cmd_model_download(REFERENCE, backend="splash")


@when("I list the combined Arkai model inventory")  # ty: ignore[call-non-callable]
def combined_inventory(context: Any) -> None:
    """Native and provider model stores appear in the same user-facing command."""
    output = StringIO()
    with redirect_stdout(output):
        model.cmd_model_list()
    context.native_output = output.getvalue()


@then("the native model reference is listed")  # ty: ignore[call-non-callable]
def native_reference_visible(context: Any) -> None:
    """Preserve the full provider-qualified model identity."""
    assert REFERENCE in context.native_output


@when("I resolve a model through the Splash store")  # ty: ignore[call-non-callable]
def native_resolve(context: Any) -> None:
    """Resolve a local installed selection without starting inference."""
    model.cmd_model_resolve(REFERENCE, backend="splash")


@when("I remove a model through the Splash store")  # ty: ignore[call-non-callable]
def native_remove(context: Any) -> None:
    """Delegate selection removal to the native lock-aware adapter."""
    model.cmd_model_remove(REFERENCE, backend="splash")


@when("I preview native Splash cache pruning")  # ty: ignore[call-non-callable]
def native_prune_preview(context: Any) -> None:
    """Request a preview, not destructive cache collection."""
    model.cmd_model_prune(dry_run=True, backend="splash")


@then('the native model operation is "{operation}"')  # ty: ignore[call-non-callable]
def native_operation_selected(context: Any, operation: str) -> None:
    """Check the operation sent to the versioned installer bridge."""
    assert context.native_operation == operation


@then("native cache pruning is a dry run")  # ty: ignore[call-non-callable]
def native_prune_non_destructive(context: Any) -> None:
    """A preview must never ask the bridge to delete files."""
    assert context.native_settings["dry_run"] is True
