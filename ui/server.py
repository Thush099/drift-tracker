"""Lightweight local web UI server for the drift-tracker batch observability report.

Serves the monitoring workbench in ``ui/index.html`` and exposes a JSON API that runs
the real ``DriftTracker.compare()``. Standard library only. Start with
``python ui/server.py`` and open the printed URL.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import DriftTracker  # noqa: E402
from src.utils import generate_demo_datasets  # noqa: E402

INDEX_HTML = (REPO_ROOT / "ui" / "index.html").read_text(encoding="utf-8")

MAX_CSV_BYTES = 2_000_000


def _read_csv(csv_text: str, label: str):
    import pandas as pd

    if not isinstance(csv_text, str) or not csv_text.strip():
        raise ValueError(f"{label} CSV must be a non-empty CSV string")
    if len(csv_text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError(f"{label} CSV exceeds {MAX_CSV_BYTES} bytes")
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise ValueError(f"{label} CSV could not be parsed: {exc}") from exc


def run_compare(
    reference_csv: Optional[str] = None,
    current_csv: Optional[str] = None,
    prediction_col: Optional[str] = None,
    proba_col: Optional[str] = None,
    target_col: Optional[str] = None,
    use_demo: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare reference vs current data and return the drift report.

    When ``use_demo`` is set (or both CSVs are omitted), generated demo datasets are
    used so the workbench works out of the box.
    """
    if use_demo or (reference_csv is None and current_csv is None):
        ref_df, cur_df = generate_demo_datasets(seed=seed)
    else:
        if reference_csv is None or current_csv is None:
            raise ValueError("reference_csv and current_csv must both be provided")
        ref_df = _read_csv(reference_csv, "reference")
        cur_df = _read_csv(current_csv, "current")

    tracker = DriftTracker(
        prediction_col=prediction_col if prediction_col is not None else "prediction",
        proba_col=proba_col if proba_col is not None else "pred_proba",
        target_col=target_col if target_col is not None else "target",
    )
    report = tracker.compare(ref_df, cur_df)
    return report.to_dict()


MAX_REQUEST_BYTES = 4_000_000


def build_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Create (but do not start) the UI HTTP server bound to ``host:port``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"status": "ok"})
            else:
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/compare":
                self._send_json({"error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_REQUEST_BYTES:
                    self._send_json(
                        {"error": f"request body exceeds {MAX_REQUEST_BYTES} bytes"},
                        status=413,
                    )
                    return
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                result = run_compare(
                    reference_csv=payload.get("reference_csv"),
                    current_csv=payload.get("current_csv"),
                    prediction_col=payload.get("prediction_col"),
                    proba_col=payload.get("proba_col"),
                    target_col=payload.get("target_col"),
                    use_demo=bool(payload.get("use_demo", False)),
                    seed=int(payload.get("seed", 42)),
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the drift-tracker observability UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port)
    print(f"Drift observability workbench running at http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())