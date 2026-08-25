"""Command-line interface: argument parsing and dispatch."""

import argparse
import sys

from aitool import __version__
from aitool import config as config_module
from aitool import engine as engine_module
from aitool import model as model_module


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

    # Engine command
    engine_parser = subparsers.add_parser("engine", help="Manage inference engine")
    engine_subparsers = engine_parser.add_subparsers(dest="engine_cmd")
    start_parser = engine_subparsers.add_parser("start", help="Start inference server")
    start_parser.add_argument("--model", help="Override model from config")
    start_parser.add_argument("--gpu-layers", type=int, help="Override GPU layers")
    start_parser.add_argument("--context", type=int, help="Override context size")
    engine_subparsers.add_parser("stop", help="Stop inference server")
    engine_subparsers.add_parser("status", help="Show inference server status")

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
        elif args.command == "engine":
            if args.engine_cmd == "start":
                engine_module.cmd_engine_start(args.model, args.gpu_layers, args.context)
            elif args.engine_cmd == "stop":
                engine_module.cmd_engine_stop()
            elif args.engine_cmd == "status":
                engine_module.cmd_engine_status()
            else:
                engine_parser.print_help()
        else:
            parser.print_help()
    except Exception as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
