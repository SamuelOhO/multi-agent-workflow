#!/usr/bin/env python
"""Server runner script.

Usage:
    python scripts/run_server.py [--mode dev|prod] [--host HOST] [--port PORT]

Examples:
    python scripts/run_server.py                    # Development mode (default)
    python scripts/run_server.py --mode prod        # Production mode
    python scripts/run_server.py --port 8080        # Custom port
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Run the server with the specified configuration."""
    parser = argparse.ArgumentParser(
        description="Run the Agent Orchestrator server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_server.py                    # Development mode
    python scripts/run_server.py --mode prod        # Production mode
    python scripts/run_server.py --host 127.0.0.1   # Localhost only
    python scripts/run_server.py --port 8080        # Custom port
    python scripts/run_server.py --workers 8        # Production with 8 workers
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Server mode: dev (with reload) or prod (with workers)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config or 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config or 8000)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (prod mode only, default: 4)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (dev mode only)",
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info for dev, warning for prod)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file",
    )

    args = parser.parse_args()

    # Import uvicorn here to allow --help without uvicorn installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    # Load configuration to get defaults
    from src.utils.config import init_config

    config_path = args.config
    if config_path is None:
        default_config = project_root / "configs" / "app.yaml"
        if default_config.exists():
            config_path = str(default_config)

    config = init_config(yaml_path=config_path, env_file=args.env_file)

    # Determine host and port
    host = args.host or config.app.host
    port = args.port or config.app.port

    # Set environment variables for app to pick up
    if args.host:
        os.environ["APP_HOST"] = args.host
    if args.port:
        os.environ["APP_PORT"] = str(args.port)

    if args.mode == "dev":
        # Development mode with hot-reload
        log_level = args.log_level or "info"
        reload = True if args.reload or args.mode == "dev" else False

        print(f"\n{'='*60}")
        print("  Agent Orchestrator - Development Server")
        print(f"{'='*60}")
        print("  Mode:      Development")
        print(f"  Host:      {host}")
        print(f"  Port:      {port}")
        print(f"  Reload:    {'enabled' if reload else 'disabled'}")
        print(f"  Log Level: {log_level}")
        print(f"  Docs:      http://{host}:{port}/docs")
        print(f"{'='*60}\n")

        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=reload,
            reload_dirs=[str(project_root / "src")],
            log_level=log_level,
        )

    else:
        # Production mode with workers
        log_level = args.log_level or "warning"
        workers = args.workers

        print(f"\n{'='*60}")
        print("  Agent Orchestrator - Production Server")
        print(f"{'='*60}")
        print("  Mode:      Production")
        print(f"  Host:      {host}")
        print(f"  Port:      {port}")
        print(f"  Workers:   {workers}")
        print(f"  Log Level: {log_level}")
        print(f"{'='*60}\n")

        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=False,
            workers=workers,
            log_level=log_level,
            access_log=False,
        )


if __name__ == "__main__":
    main()
