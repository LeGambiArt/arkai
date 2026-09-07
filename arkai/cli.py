"""Command-line interface: argument parsing and dispatch."""

import argparse
import sys
import traceback

from arkai import __version__, utils


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

    # modules order define order of commands in the help message
    modules = [
        "status",
        "config",
        "agent",
        "inference",
        "model",
        "wtmcp",
        "sandbox",
        "vectordb",
        "rag",
        "benchmark",
    ]
    for module in modules:
        module_name = f"arkai.{module}"
        if module_name not in sys.modules:
            __import__(module_name)
        getattr(sys.modules[module_name], "ingest_cli_options")(subparsers)

    args = parser.parse_args()

    try:
        cmd = getattr(sys.modules[f"arkai.{args.command}"], "exec_cmd")
        cmd(args)
        # cmd(args, parser=subparser_pool.get(args.command))
    except Exception as e:
        traceback.print_exc()
        _, _, exc_tb = sys.exc_info()
        fname, lineno, fn, _ = traceback.extract_tb(exc_tb, 1)[-1]
        utils.error(f"fatal: {e}\n\tat {fn} ({fname}:{str(lineno).strip()})")
        sys.exit(1)
