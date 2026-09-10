"""Keep an Arkai-managed wtmcp server attached to a live parent process."""

import argparse
import os
import signal
import subprocess
import sys


def run(command: list[str], child_pid_path: str) -> int:
    """Run wtmcp and forward termination signals until it exits.

    Args:
        command: wtmcp command and arguments to execute.
        child_pid_path: File used to publish the wtmcp child PID.

    Returns:
        The wtmcp process exit status.
    """
    os.makedirs(os.path.dirname(child_pid_path), exist_ok=True)
    child = subprocess.Popen(command)
    with open(child_pid_path, "w", encoding="utf-8") as pid_file:
        pid_file.write(str(child.pid))

    def terminate_child(_signum: int, _frame: object) -> None:
        """Request a graceful shutdown of the wtmcp child process."""
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, terminate_child)
    signal.signal(signal.SIGINT, terminate_child)

    try:
        return child.wait()
    finally:
        try:
            os.remove(child_pid_path)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    """Parse supervisor arguments and run the wtmcp child process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-pid-path", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a wtmcp command is required")

    return run(command, args.child_pid_path)


if __name__ == "__main__":
    sys.exit(main())
