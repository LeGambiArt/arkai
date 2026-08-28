"""Command-line interface: argument parsing and dispatch."""

import argparse
import sys

from arkai import __version__
from arkai import agent as agent_module
from arkai import config as config_module
from arkai import engine as engine_module
from arkai import model as model_module
from arkai import sandbox as sandbox_module
from arkai import wtmcp as wtmcp_module


def _add_agent_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by agent start and agent prompt.

    Args:
        parser: The argparse subparser to add arguments to
    """
    parser.add_argument("-a", "--agent", help="Override agent")
    parser.add_argument("-m", "--model", help="Override model")
    parser.add_argument(
        "-I",
        "--no-inference",
        action="store_true",
        help="Do not start inference engine server",
    )
    parser.add_argument("-M", "--no-mcp", action="store_true", help="Skip wtmcp initialization")
    parser.add_argument("--no-sandbox", action="store_true", help="Skip arapuca sandbox")
    parser.add_argument(
        "-s",
        "--sandbox",
        metavar="PROFILE",
        help="Use specific sandbox profile for this run",
    )
    parser.add_argument(
        "-v",
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume in the sandbox (format: /path or /path:ro)",
    )
    parser.add_argument(
        "-e",
        "--env",
        action="append",
        dest="environment",
        metavar="KEY=VALUE",
        help="Set an environment variable in the sandbox (KEY=VALUE). Can be used multiple times",
    )
    cwd_group = parser.add_mutually_exclusive_group()
    cwd_group.add_argument(
        "--no-cwd", action="store_true", help="Do not mount the current directory in the sandbox"
    )
    cwd_group.add_argument(
        "--cwd", metavar="PATH", help="Override the directory mounted as cwd in the sandbox"
    )


def main():
    """Parse arguments and dispatch to command handlers."""
    parser = argparse.ArgumentParser(
        prog="arkai",
        description="Local AI inference workflow management",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_cmd")
    validate_parser = config_subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--file", help="Config file to validate (default: .arkai.yaml)")
    config_subparsers.add_parser("init", help="Initialize configuration file")

    # Model command
    model_parser = subparsers.add_parser("model", help="Manage models")
    model_subparsers = model_parser.add_subparsers(dest="model_cmd")
    download_parser = model_subparsers.add_parser(
        "download", help="Download model from HuggingFace"
    )
    download_parser.add_argument("hf_repo", help="HuggingFace repo ID")
    model_subparsers.add_parser("list", help="List available models")
    remove_parser = model_subparsers.add_parser("remove", help="Remove model")
    remove_parser.add_argument("model_name", help="Model file name")
    convert_parser = model_subparsers.add_parser(
        "convert", help="Convert HuggingFace model to GGUF format"
    )
    convert_parser.add_argument("model", help="HuggingFace model ID or path")
    convert_parser.add_argument(
        "-q", "--quantization", default="Q6_K", help="Quantization level (default: Q6_K)"
    )
    convert_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: ~/.local/share/arkai/models/MODEL-QUANTIZATION.gguf)",
    )

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Manage inference engine server")
    inference_subparsers = inference_parser.add_subparsers(dest="inference_cmd")
    start_parser = inference_subparsers.add_parser("start", help="Start inference server")
    start_parser.add_argument("--model", help="Override model from config")
    start_parser.add_argument("--gpu-layers", type=int, help="Override GPU layers")
    start_parser.add_argument("--context", type=int, help="Override context size")
    start_parser.add_argument("--port", type=int, help="Override port from config")
    inference_subparsers.add_parser("stop", help="Stop inference server")
    inference_subparsers.add_parser("status", help="Show inference server status")

    # wtmcp command
    wtmcp_parser = subparsers.add_parser("wtmcp", help="Manage wtmcp plugins and server")
    wtmcp_subparsers = wtmcp_parser.add_subparsers(dest="wtmcp_cmd")
    list_parser = wtmcp_subparsers.add_parser("list", help="List available plugins")
    list_parser.add_argument(
        "--port", type=int, help="Port of running instance to show plugins for"
    )
    enable_parser = wtmcp_subparsers.add_parser("enable", help="Enable a plugin")
    enable_parser.add_argument("plugin", help="Plugin name")
    disable_parser = wtmcp_subparsers.add_parser("disable", help="Disable a plugin")
    disable_parser.add_argument("plugin", help="Plugin name")
    start_parser = wtmcp_subparsers.add_parser("start", help="Start wtmcp server")
    start_parser.add_argument("--path", help="Override wtmcp binary path from config")
    start_parser.add_argument("--port", type=int, help="Override port from config")
    start_parser.add_argument(
        "--enable",
        action="append",
        dest="enable_plugins",
        help="Enable plugin (can be used multiple times)",
    )
    start_parser.add_argument(
        "--disable",
        action="append",
        dest="disable_plugins",
        help="Disable plugin (can be used multiple times)",
    )
    stop_parser = wtmcp_subparsers.add_parser("stop", help="Stop wtmcp server")
    stop_parser.add_argument("--port", type=int, help="Port of instance to stop")
    status_parser = wtmcp_subparsers.add_parser("status", help="Show wtmcp server status")
    status_parser.add_argument(
        "--port", type=int, help="Port to show status for (or all if not specified)"
    )

    # Sandbox command
    sandbox_parser = subparsers.add_parser("sandbox", help="Manage sandbox profiles")
    sandbox_subparsers = sandbox_parser.add_subparsers(dest="sandbox_cmd")

    list_parser = sandbox_subparsers.add_parser("list", help="List sandbox profiles")
    list_parser = list_parser  # noqa: F841 - keep reference for consistency

    show_parser = sandbox_subparsers.add_parser("show", help="Show sandbox profile details")
    show_parser.add_argument("profile_name", help="Profile name to show (or 'default'/'active')")

    create_parser = sandbox_subparsers.add_parser("create", help="Create sandbox profile")
    create_parser.add_argument("name", help="Profile name")
    create_parser.add_argument("--from", dest="from_profile", help="Base profile to copy from")
    create_parser.add_argument("--path", help="Path to arapuca binary")
    create_parser.add_argument("--memory", type=int, help="Memory in MB")
    create_parser.add_argument("--cpus", type=int, help="CPU count")
    create_parser.add_argument("--pids", type=int, help="PID limit")
    create_parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    create_parser.add_argument(
        "-v",
        "--volume",
        action="append",
        dest="volumes",
        help="Mount a volume in the sandbox (format: /path or /path:ro)",
    )
    create_parser.add_argument(
        "-e",
        "--env",
        action="append",
        dest="environment",
        metavar="KEY=VALUE",
        help="Set an environment variable in the sandbox (KEY=VALUE). Can be used multiple times",
    )

    delete_parser = sandbox_subparsers.add_parser("delete", help="Delete sandbox profile")
    delete_parser.add_argument("name", help="Profile name")

    setdef_parser = sandbox_subparsers.add_parser("set-default", help="Set profile as default")
    setdef_parser.add_argument("name", help="Profile name")

    active_parser = sandbox_subparsers.add_parser("active", help="Set active profile")
    active_parser.add_argument("profile_name", nargs="?", help="Profile name (or empty to clear)")

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Manage interactive agent")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_cmd")

    agent_start_parser = agent_subparsers.add_parser("start", help="Start interactive agent")
    _add_agent_common_args(agent_start_parser)

    agent_prompt_parser = agent_subparsers.add_parser(
        "prompt", help="Run agent with a prompt non-interactively"
    )
    _add_agent_common_args(agent_prompt_parser)
    agent_prompt_parser.add_argument(
        "-o", "--output", metavar="FILE", help="Write agent output to file instead of stdout"
    )
    agent_prompt_parser.add_argument(
        "prompt_text", nargs="*", help="Prompt text (reads from stdin if not provided)"
    )

    args = parser.parse_args()

    try:
        if args.command == "config":
            if args.config_cmd == "validate":
                config_module.cmd_config_validate(args.file)
            elif args.config_cmd == "init":
                config_module.cmd_config_init()
            else:
                config_parser.print_help()
        elif args.command == "model":
            if args.model_cmd == "list":
                model_module.cmd_model_list()
            elif args.model_cmd == "download":
                model_module.cmd_model_download(args.hf_repo)
            elif args.model_cmd == "remove":
                model_module.cmd_model_remove(args.model_name)
            elif args.model_cmd == "convert":
                model_module.cmd_model_convert(args.model, args.quantization, args.output)
            else:
                model_parser.print_help()
        elif args.command == "inference":
            if args.inference_cmd == "start":
                engine_module.cmd_engine_start(args.model, args.gpu_layers, args.context, args.port)
            elif args.inference_cmd == "stop":
                engine_module.cmd_engine_stop()
            elif args.inference_cmd == "status":
                engine_module.cmd_engine_status()
            else:
                inference_parser.print_help()
        elif args.command == "wtmcp":
            if args.wtmcp_cmd == "list":
                wtmcp_module.cmd_wtmcp_list(args.port)
            elif args.wtmcp_cmd == "enable":
                wtmcp_module.cmd_wtmcp_enable(args.plugin)
            elif args.wtmcp_cmd == "disable":
                wtmcp_module.cmd_wtmcp_disable(args.plugin)
            elif args.wtmcp_cmd == "start":
                wtmcp_module.cmd_wtmcp_start(
                    args.path,
                    args.port,
                    args.enable_plugins,
                    args.disable_plugins,
                )
            elif args.wtmcp_cmd == "stop":
                wtmcp_module.cmd_wtmcp_stop(args.port)
            elif args.wtmcp_cmd == "status":
                wtmcp_module.cmd_wtmcp_status(args.port)
            else:
                wtmcp_parser.print_help()
        elif args.command == "sandbox":
            if args.sandbox_cmd == "create":
                create_env = (
                    {k: v for e in args.environment for k, _, v in [e.partition("=")]}
                    if args.environment
                    else None
                )
                sandbox_module.cmd_sandbox_create(
                    args.name,
                    args.from_profile,
                    args.path,
                    args.memory,
                    args.cpus,
                    args.pids,
                    args.timeout,
                    args.volumes,
                    create_env,
                )
            elif args.sandbox_cmd == "list":
                sandbox_module.cmd_sandbox_list()
            elif args.sandbox_cmd == "show":
                sandbox_module.cmd_sandbox_show(args.profile_name)
            elif args.sandbox_cmd == "delete":
                sandbox_module.cmd_sandbox_delete(args.name)
            elif args.sandbox_cmd == "set-default":
                sandbox_module.cmd_sandbox_set_default(args.name)
            elif args.sandbox_cmd == "active":
                sandbox_module.cmd_sandbox_active(args.profile_name)
            else:
                sandbox_parser.print_help()
        elif args.command == "agent":
            if args.agent_cmd in ("start", "prompt"):
                agent_env = (
                    {k: v for e in args.environment for k, _, v in [e.partition("=")]}
                    if args.environment
                    else None
                )
                if args.agent_cmd == "start":
                    agent_module.cmd_agent(
                        args.agent,
                        args.model,
                        args.no_inference,
                        # args.keep_mcp,
                        args.no_mcp,
                        args.no_sandbox,
                        args.no_cwd,
                        args.cwd,
                        args.sandbox,
                        args.volumes,
                        agent_env,
                    )
                else:
                    agent_module.cmd_agent_prompt(
                        args.prompt_text,
                        args.agent,
                        args.model,
                        args.no_inference,
                        # args.keep_mcp,
                        args.no_mcp,
                        args.no_sandbox,
                        args.no_cwd,
                        args.cwd,
                        args.sandbox,
                        args.volumes,
                        agent_env,
                        args.output,
                    )
            else:
                agent_parser.print_help()
        else:
            parser.print_help()
    except Exception as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
