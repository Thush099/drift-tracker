"""Tests for the drift-tracker observability UI server (ui/server.py)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ui.server import build_server, run_compare

REFERENCE_CSV = "id,feature,prediction,pred_proba,target\n1,0.5,1,0.9,1\n2,0.4,0,0.2,0\n3,0.6,1,0.8,1\n"
CURRENT_CSV = "id,feature,prediction,pred_proba,target\n1,9.5,1,0.9,1\n2,9.4,0,0.2,0\n3,9.6,1,0.8,1\n"


@pytest.fixture()
def server():
    srv = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _post(base: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}/api/compare",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request)
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_demo_comparison_runs_real_tracker(server):
    status, payload = _post(server, {"use_demo": True})
    assert status == 200
    assert payload["overall_status"] in ("OK", "WARN", "ALERT")
    assert payload["reference_rows"] > 0 and payload["current_rows"] > 0
    assert payload["numeric_features"] or payload["categorical_features"]


def test_uploaded_csvs_flag_the_injected_shift(server):
    status, payload = _post(
        server,
        {"reference_csv": REFERENCE_CSV, "current_csv": CURRENT_CSV},
    )
    assert status == 200
    feature = next(f for f in payload["numeric_features"] if f["column"] == "feature")
    assert feature["psi"] > 0.5
    assert feature["status"] in ("WARN", "ALERT")


def test_health_and_index_served(server):
    assert json.loads(urllib.request.urlopen(server + "/api/health").read())["status"] == "ok"
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "observability" in html and "/api/compare" in html


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"reference_csv": "a\n1\n"}, "reference_csv and current_csv must both be provided"),
        ({"reference_csv": "", "current_csv": "a\n1\n"}, "reference CSV must be a non-empty"),
        ({"reference_csv": 'a\n"unclosed', "current_csv": "a\n1\n"}, "reference CSV could not be parsed"),
    ],
)
def test_invalid_inputs_rejected_with_clear_message(server, payload, message):
    status, body = _post(server, payload)
    assert status == 400
    assert message in body["error"]


def test_run_compare_defaults_to_demo():
    result = run_compare()
    assert "overall_status" in result

def test_index_has_accessibility_features(server):
    """The served page ships skip-link navigation, a main landmark, and reduced-motion guard."""
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert 'class="skip-link"' in html, "missing skip link for keyboard users"
    assert '<main class="layout">' in html, "missing main landmark"
    assert 'role="alert"' in html, "error box must announce to assistive tech"
    assert "prefers-reduced-motion" in html, "missing reduced-motion guard"
