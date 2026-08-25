"""Command-line interface: argument parsing and dispatch."""

import sys
import argparse
from aitool import __version__, config as config_module


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
    validate_parser = config_subparsers.add_parser(
        "validate", help="Validate configuration"
    )
    validate_parser.add_argument(
        "--file", help="Config file to validate (default: .aitool.yaml)"
    )
    config_subparsers.add_parser("init", help="Initialize configuration file")

    args = parser.parse_args()

    if args.command == "config":
        if args.config_cmd == "validate":
            config_module.cmd_config_validate(args.file)
        elif args.config_cmd == "init":
            config_module.cmd_config_init()
        else:
            config_parser.print_help()


if __name__ == "__main__":
    main()
