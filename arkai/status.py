"""Status monitoring for all servers (inference, wtmcp, vectordb)."""

from arkai import engine as engine_module
from arkai import utils
from arkai import vectordb as vectordb_module
from arkai import wtmcp as wtmcp_module


def cmd_status() -> None:
    """Show status of all servers (inference, wtmcp, vectordb)."""
    utils.info("=== arkai Server Status ===\n")

    # Check inference server
    inference_running = engine_module.is_inference_running()
    if inference_running:
        pid = utils.read_pid(engine_module.get_inference_pid_path())
        utils.info(f"✓ Inference: running (PID {pid})")
        try:
            state = utils.load_yaml(engine_module.get_inference_state_path())
            port = state.get("port", 8081)
            model = state.get("model", "unknown")
            utils.info(f"  Model: {model}")
            utils.info(f"  Port: {port}\n")
        except Exception:
            utils.info("  (state file error)\n")
    else:
        utils.info("✗ Inference: stopped\n")

    # Check wtmcp server(s)
    running_ports = wtmcp_module._get_running_instances()
    if running_ports:
        utils.info(f"✓ wtmcp: running ({len(running_ports)} instance(s))")
        for port in running_ports:
            pid = utils.read_pid(wtmcp_module.get_wtmcp_pid_path(port))
            utils.info(f"  Port {port} (PID {pid})")
        utils.info("")
    else:
        utils.info("✗ wtmcp: stopped\n")

    # Check vectordb server
    vectordb_running = vectordb_module.is_vectordb_running()
    if vectordb_running:
        pid = utils.read_pid(vectordb_module.get_vectordb_pid_path())
        utils.info(f"✓ Vectordb: running (PID {pid})")
        try:
            state = utils.load_yaml(vectordb_module.get_vectordb_state_path())
            port = state.get("port", 8082)
            databases = state.get("databases", [])
            utils.info(f"  Port: {port}")
            if databases:
                utils.info(f"  Databases: {', '.join(databases)}\n")
            else:
                utils.info("  Databases: (none)\n")
        except Exception:
            utils.info("  (state file error)\n")
    else:
        utils.info("✗ Vectordb: stopped\n")
