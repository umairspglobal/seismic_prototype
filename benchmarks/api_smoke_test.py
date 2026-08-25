"""End-to-end smoke test of the inference API used by the React client.

Requires the server to be running: uvicorn server.main:app --port 8000
Exercises file listing, slice image fetch, prepare, warm point clicks,
and (optionally) a streamed propagation.

Usage:
    python benchmarks/api_smoke_test.py [--propagate]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path: str) -> bytes:
    with urllib.request.urlopen(BASE + path, timeout=300) as res:
        return res.read()


def post(path: str, payload: dict, stream: bool = False):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    res = urllib.request.urlopen(req, timeout=600)
    if stream:
        return res
    return json.loads(res.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propagate", action="store_true")
    args = parser.parse_args()

    files = json.loads(get("/api/files"))
    assert files, "No seismic files served"
    volume = next((f for f in files if f["kind"] == "3d"), files[0])
    name = volume["name"]
    axis, index = "inline", volume["axes"]["inline"] // 2
    print(f"file={name} axis={axis} index={index}")

    t0 = time.perf_counter()
    image = get(f"/api/slice?file={name}&axis={axis}&index={index}")
    print(f"slice image: {len(image)} bytes in {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    prep = post("/api/prepare", {"file": name, "axis": axis, "index": index})
    print(f"prepare: {time.perf_counter() - t0:.3f}s (encoder {prep['prepare_seconds']:.3f}s)")

    # Emulate a burst of interactive clicks after the slice is prepared.
    shape = volume["shape"]
    col, row = shape[1] // 2, shape[2] // 2
    click_times = []
    for i in range(5):
        payload = {
            "file": name,
            "axis": axis,
            "index": index,
            "points": [[col + i, row]],
            "labels": [1],
        }
        t0 = time.perf_counter()
        result = post("/api/segment", payload)
        click_times.append(time.perf_counter() - t0)
        print(
            f"click {i + 1}: {click_times[-1] * 1000:.0f} ms end-to-end "
            f"(decoder {result['timings'].get('prompt_decode', 0) * 1000:.0f} ms, "
            f"coverage {100 * result['coverage']:.1f}%)"
        )
    print(
        f"warm click p50={statistics.median(click_times) * 1000:.0f} ms "
        f"max={max(click_times) * 1000:.0f} ms"
    )

    if args.propagate:
        # Two objects at once: exercises the shared-session multi-object path.
        payload = {
            "file": name,
            "axis": axis,
            "index": index,
            "objects": [
                {"id": 0, "points": [[col, row]], "labels": [1]},
                {"id": 1, "points": [[col // 2, row]], "labels": [1]},
            ],
        }
        t0 = time.perf_counter()
        first_frame_at = None
        events = 0
        res = post("/api/propagate", payload, stream=True)
        for raw in res:
            event = json.loads(raw)
            if event["type"] == "frame":
                events += 1
                if first_frame_at is None:
                    first_frame_at = time.perf_counter() - t0
            elif event["type"] == "done":
                total = time.perf_counter() - t0
                print(
                    f"propagation: {events} frames streamed, first frame at "
                    f"{first_frame_at:.2f}s, total {total:.1f}s "
                    f"({event['timings'].get('fps', 0):.2f} fps)"
                )
            elif event["type"] == "error":
                raise SystemExit(f"propagation error: {event['message']}")

    print("smoke test passed")


if __name__ == "__main__":
    main()
