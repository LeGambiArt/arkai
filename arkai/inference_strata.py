"""Strata CLI-backed inference service integration."""

import json
from typing import Any

from arkai import utils
from arkai.inference_backend import SamplingSettings


class StrataBackend:
    """Manage Strata's background service through its supported CLI."""

    name = "strata"

    def check_environment(self) -> None:
        """The configured CLI is resolved when building its command."""

    def build_command(
        self,
        path: str,
        model: str,
        port: int,
        gpu_layers: int,
        context_size: int,
        path_is_configured: bool = False,
        sampling: SamplingSettings | None = None,
    ) -> list[str]:
        """Build a local-only ``strata launch`` command."""
        if sampling:
            raise RuntimeError(
                "Strata backend does not support Arkai server sampling defaults; "
                "remove inference sampling settings"
            )
        if gpu_layers != -1:
            raise RuntimeError(
                "Strata selects GPU execution automatically; remove inference.gpu_layers"
            )

        binary = utils.resolve_binary(path, search_path=not path_is_configured)
        return [
            binary,
            "launch",
            "--model",
            model,
            "--context",
            str(context_size),
            "--access",
            "local",
            "--port",
            str(port),
            "--tune",
            "skip",
            "--no-tui",
            "--output",
            "json",
        ]

    def build_plan_command(self, command: list[str]) -> list[str]:
        """Return the side-effect-free plan variant of a launch command."""
        return [*command[:2], "--plan", *command[2:]]

    def build_status_command(self, path: str) -> list[str]:
        """Build Strata's machine-readable launch status query."""
        binary = utils.resolve_binary(path, search_path=False)
        return [binary, "launch", "--status", "--output", "json"]

    def build_stop_command(self, path: str, profile: str) -> list[str]:
        """Build a stop command scoped to one Strata launch profile."""
        if not profile.startswith("launch-") or not profile.removeprefix("launch-"):
            raise RuntimeError(f"Invalid Strata launch profile: {profile!r}")
        binary = utils.resolve_binary(path, search_path=False)
        return [binary, "launch", "--stop", "--preset", profile.removeprefix("launch-")]

    @staticmethod
    def parse_json_output(output: str, description: str) -> dict[str, Any]:
        """Parse Strata's JSON response, allowing a trailing human-readable status line."""
        try:
            result = json.loads(output)
        except (TypeError, ValueError):
            try:
                result, _ = json.JSONDecoder().raw_decode(output.lstrip())
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Strata returned invalid JSON while {description}") from error
        if not isinstance(result, dict):
            raise RuntimeError(f"Strata returned an invalid response while {description}")
        return result

    @classmethod
    def profile_is_running(cls, status_output: str, profile: str) -> bool:
        """Return whether one named Strata launch is starting or ready."""
        result = cls.parse_json_output(status_output, "checking service status")
        if result.get("ok") is not True or not isinstance(result.get("running"), list):
            raise RuntimeError("Strata returned an invalid launch status response")
        return any(
            isinstance(item, dict)
            and item.get("profile") == profile
            and item.get("state") in {"ready", "starting"}
            for item in result["running"]
        )

    @classmethod
    def running_launch_for_port(cls, status_output: str, port: int) -> dict[str, Any] | None:
        """Return a ready/starting Strata profile serving the requested port."""
        result = cls.parse_json_output(status_output, "checking service status")
        running = result.get("running")
        if result.get("ok") is not True or not isinstance(running, list):
            raise RuntimeError("Strata returned an invalid launch status response")
        return next(
            (
                item
                for item in running
                if isinstance(item, dict)
                and item.get("port") == port
                and item.get("state") in {"ready", "starting"}
            ),
            None,
        )
