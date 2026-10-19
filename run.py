"""Run the local Phase-2 Flask demonstration: python run.py."""
import logging
import os

from waitress import serve
from webapp import create_app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.environ.get("WEB_PORT", "5000"))
    app = create_app()
    print(f"Phase-2 research demonstration: http://127.0.0.1:{port}", flush=True)
    print("The model loads once on the first prediction. Press Ctrl+C to stop.", flush=True)
    serve(app, host="127.0.0.1", port=port, threads=4,
          max_request_body_size=app.config["MAX_CONTENT_LENGTH"])


if __name__ == "__main__":
    main()
