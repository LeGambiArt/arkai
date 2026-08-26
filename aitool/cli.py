"""Command-line interface: argument parsing and dispatch."""

import argparse
import sys

from aitool import __version__
from aitool import agent as agent_module
from aitool import config as config_module
from aitool import engine as engine_module
from aitool import model as model_module
from aitool import wtmcp as wtmcp_module


def main():
    """Parse arguments and dispatch to command handlers."""
    parser = argparse.ArgumentParser(
        prog="aitool",
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
    validate_parser.add_argument("--file", help="Config file to validate (default: .aitool.yaml)")
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
        help="Output file path (default: ~/.local/share/aitool/models/MODEL-QUANTIZATION.gguf)",
    )

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Manage inference engine")
    inference_subparsers = inference_parser.add_subparsers(dest="inference_cmd")
    start_parser = inference_subparsers.add_parser("start", help="Start inference server")
    start_parser.add_argument("--model", help="Override model from config")
    start_parser.add_argument("--gpu-layers", type=int, help="Override GPU layers")
    start_parser.add_argument("--context", type=int, help="Override context size")
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

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Start interactive agent")
    agent_parser.add_argument("--agent", help="Override agent")
    agent_parser.add_argument("--model", help="Override model")
    agent_parser.add_argument(
        "-I",
        "--keep-inference",
        action="store_true",
        help="Keep inference server running after exit",
    )
    agent_parser.add_argument(
        "-M", "--keep-mcp", action="store_true", help="Keep wtmcp server running after exit"
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
                engine_module.cmd_engine_start(args.model, args.gpu_layers, args.context)
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
        elif args.command == "agent":
            agent_module.cmd_agent(args.agent, args.model, args.keep_inference, args.keep_mcp)
        else:
            parser.print_help()
    except Exception as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
