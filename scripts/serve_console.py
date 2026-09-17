"""Serve the Support Intelligence Console. Never initializes an API provider."""
import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.console.api import ConsoleService, make_server


def get_server_config():
    """Determine host and port based on environment and local defaults."""
    if "PORT" in os.environ:
        default_port = int(os.environ["PORT"])
        default_host = os.environ.get("HOST", "0.0.0.0")
    else:
        default_port = 8765
        default_host = os.environ.get("HOST", "127.0.0.1")
    return default_host, default_port


def main():
    default_host, default_port = get_server_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--host", type=str, default=default_host)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    service = ConsoleService()
    try:
        service.initialize()
    except Exception:
        print("Frozen artifacts unavailable. The console will show an unavailable state.")
    server = make_server(args.port, service, host=args.host)
    print(f"Support Intelligence Console: http://{args.host}:{args.port} (offline generator)", flush=True)
    print("Open the URL above for both frontend and /api/*; a separate python -m http.server cannot serve the API.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
