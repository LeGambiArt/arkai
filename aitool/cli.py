"""Command-line interface: argument parsing and dispatch."""

import sys
import argparse
from aitool import __version__


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

    parser.add_argument(
        "-h", "--help",
        action="help",
        help="show this help message and exit",
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
