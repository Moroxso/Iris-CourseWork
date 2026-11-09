"""Local presentation, ordering, and read-only transport boundaries."""

import importlib.util
import json
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest


def viewer() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts/coursework_live_viewer.py"
    spec = importlib.util.spec_from_file_location("coursework_live_viewer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_order_and_request_reset() -> None:
    state = viewer().LiveState()
    snapshot = lambda: json.loads(state.snapshot())  # noqa: E731
    assert snapshot()["stage"] == "ready"
    state.publish("request", {"request": 1})
    state.publish("first-pass", {"prompt": "<script>payload</script>"})
    before = snapshot()
    with pytest.raises(ValueError, match="ordered"):
        state.publish("cycle", {"row": {"cycle": 2}, "cycle_count": 3})
    with pytest.raises(ValueError, match="complete workspace"):
        state.publish("generation", {"latents": {"source_cycle": 3}})
    assert snapshot() == before
    for n in range(1, 4):
        row = {
            "cycle": n,
            "selected_modules": [1, 4],
            "write_modules": [4, 1],
            "selected_slots": [2, 0],
            "slot_sources": [1, -1, 4],
        }
        state.publish("cycle", {"row": row, "cycle_count": 3})
        assert snapshot()["cycles"][-1] == row
        row["cycle"] = 999
        assert [c["cycle"] for c in snapshot()["cycles"]] == list(range(1, n + 1))
    state.publish("generation", {"latents": {"source_cycle": 3}})
    state.publish("completed", {"text": "answer", "stop_reason": "stop_token"})
    assert snapshot()["answer"] == "answer"
    assert snapshot()["latents"] == {"source_cycle": 3}
    state.publish("request", {"request": 2})
    state.publish("rejected", {"message": "too long"})
    assert snapshot()["request"] == 2 and snapshot()["stage"] == "rejected"
    assert snapshot()["cycles"] == [] and snapshot()["latents"] is None
    assert snapshot()["answer"] == snapshot()["prompt"] == ""
    state.publish("request", {"request": 2})
    assert snapshot()["stage"] == "request" and snapshot()["detail"] == ""


@pytest.mark.parametrize("write,slots", [([1, 4], [2, 0]), ([4, 4], [2, 0]), ([4, 1], [2, 2])])
def test_live_pairs_reject_misassociation(write: list[int], slots: list[int]) -> None:
    state = viewer().LiveState()
    state.publish("request", {"request": 1})
    before = state.snapshot()
    with pytest.raises(ValueError, match="write pairs"):
        state.publish(
            "cycle",
            {
                "row": {
                    "cycle": 1,
                    "selected_modules": [1, 4],
                    "write_modules": write,
                    "selected_slots": slots,
                    "slot_sources": [1, -1, 4],
                },
                "cycle_count": 3,
            },
        )
    assert state.snapshot() == before


def test_winning_vectors_must_match_observed_writes() -> None:
    module = viewer()
    row = {"write_modules": [4], "selected_slots": [1], "slot_norms": [0, 2.0]}
    pair = {"module": 4, "slot": 1, "proposal": [3.0] * 256, "stored": [2.0] + [0.0] * 255}
    module.validate_vectors(row, [pair])
    with pytest.raises(ValueError, match="pair mismatch"):
        module.validate_vectors(row, [dict(pair, slot=0)])
    with pytest.raises(ValueError, match="coordinates"):
        module.validate_vectors(row, [dict(pair, proposal=[float("nan")] * 256)])
    with pytest.raises(ValueError, match="norm"):
        module.validate_vectors(row, [dict(pair, stored=[0.0] * 256)])


def test_only_secret_get_routes_and_loopback() -> None:
    module = viewer()
    with module.live_view(module.LiveState()) as url:
        parts = urlsplit(url)
        assert parts.hostname == "127.0.0.1" and parts.port is not None

        def request(
            method: str, path: str, headers: dict[str, str] | None = None
        ) -> tuple[int, bytes, dict[str, str]]:
            conn = HTTPConnection("127.0.0.1", parts.port, timeout=2)
            try:
                conn.request(method, path, headers=headers or {})
                response = conn.getresponse()
                return response.status, response.read(), dict(response.getheaders())
            finally:
                conn.close()

        home = parts.path
        status, html, headers = request("GET", home)
        assert status == 200 and b"Latent embeddings" in html
        assert b".textContent=" in html and b".innerHTML" not in html
        assert headers["Cache-Control"] == "no-store"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert not any(name.startswith("Access-Control-") for name in headers)
        status, data, _ = request("GET", home + "state")
        assert status == 200 and json.loads(data)["stage"] == "ready"
        assert request("GET", "/state")[0] == 404
        assert request("GET", home + "../state")[0] == 404
        assert request("GET", home + "state?extra=1")[0] == 404
        assert request("POST", home + "state")[0] == 501
        assert request("GET", home + "state", {"Host": "example.org"})[0] == 403
        assert request("GET", home + "state", {"Sec-Fetch-Site": "cross-site"})[0] == 403

    stopped = HTTPConnection("127.0.0.1", parts.port, timeout=0.5)
    try:
        with pytest.raises(OSError):
            stopped.connect()
    finally:
        stopped.close()
