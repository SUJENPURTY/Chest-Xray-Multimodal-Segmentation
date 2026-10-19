"""Flask application factory. Importing this package never loads model weights."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from .services.inference_service import Phase2Service

ROOT = Path(__file__).resolve().parents[1]


def create_app(config=None, *, service=None):
    app = Flask(__name__)
    image_limit = int(os.environ.get("WEB_MAX_UPLOAD_MB", "12")) * 1024**2
    app.config.from_mapping(
        PROJECT_ROOT=ROOT,
        TEMP_UPLOAD_ROOT=ROOT / "outputs/webapp/tmp",
        MAX_IMAGE_BYTES=image_limit,
        # Image bytes and request bytes are different limits. Reserve space for
        # the bounded prompt, filename and multipart headers without reducing the
        # advertised image allowance. Waitress uses this same request limit.
        MAX_CONTENT_LENGTH=image_limit + 64 * 1024,
        MAX_IMAGE_PIXELS=int(os.environ.get("WEB_MAX_IMAGE_PIXELS", "16000000")),
        MAX_FORM_MEMORY_SIZE=64 * 1024,
        MAX_FORM_PARTS=8,
        PHASE2_DEVICE=os.environ.get("PHASE2_DEVICE", "auto"),
        TRUSTED_HOSTS=os.environ.get("WEB_TRUSTED_HOSTS", "127.0.0.1,localhost,[::1]").split(","),
    )
    if config:
        app.config.update(config)
    if any(app.config[key] <= 0 for key in ("MAX_IMAGE_BYTES", "MAX_CONTENT_LENGTH", "MAX_IMAGE_PIXELS")):
        raise ValueError("Upload limits must be positive")
    app.extensions["phase2_service"] = service or Phase2Service(
        Path(app.config["PROJECT_ROOT"]), app.config["PHASE2_DEVICE"],
        ROOT / "outputs/webapp/cache")
    from .routes import bp
    app.register_blueprint(bp)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if request.path.startswith("/api/") or request.path == "/":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {400: "The request is invalid.", 403: "This request is not permitted.",
                    404: "The requested endpoint was not found.", 405: "Method not allowed.",
                    413: "The upload exceeds the configured size limit.",
                    415: "Upload a supported grayscale PNG or JPEG."}
        return jsonify(error={"code": "http_error", "message": messages.get(error.code, "Request failed.")}), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        logging.getLogger(__name__).exception("Unhandled application error")
        return jsonify(error={"code": "internal_error", "message": "The request could not be completed. Check the local server log."}), 500

    return app
