"""Serve the local Support Decision Console. Never initializes an API provider."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.console.api import ConsoleService, make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    service = ConsoleService()
    try:
        service.initialize()
    except Exception:
        print("Frozen artifacts unavailable. The console will show an unavailable state.")
    server = make_server(args.port, service)
    print(f"Support Decision Console: http://127.0.0.1:{args.port} (offline generator)", flush=True)
    print("Open the URL above for both frontend and /api/*; a separate python -m http.server cannot serve the API.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
