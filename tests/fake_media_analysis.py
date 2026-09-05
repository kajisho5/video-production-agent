"""Fake media-analysis-skill process for adapter protocol tests: speaks the contract@1 / response@1 protocol
(contract --json, doctor --json, run - --json) with canned data, and can misbehave on request via
FAKE_MA_MODE = ok | empty | text | two_docs | wrong_schema | wrong_skill | wrong_version | wrong_kind | no_observation |
bad_source | error_result | hang | crash. Never runs ffmpeg. Test double only."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE.parent / "src" / "video_agent" / "tools" / "media_analysis" / "contract_0.1.0.json").read_text(encoding="utf-8"))
MODE = os.environ.get("FAKE_MA_MODE", "ok")
CALLS_LOG = os.environ.get("FAKE_MA_CALLS")


def log(kind: str) -> None:
    if CALLS_LOG:
        with open(CALLS_LOG, "a", encoding="utf-8") as fh:
            fh.write(kind + "\n")


def observation(req: dict) -> dict:
    kind = req["kind"]
    tool = CONTRACT["kind_to_tool"][kind]
    data = {
        "media_probe": {"container": {"format": "mov,mp4,m4a,3gp,3g2,mj2", "duration": 16.0, "size": 1000, "bitrate": 500, "start_time": 0.0},
                        "video": {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0, "pixel_format": "yuv420p"},
                        "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2, "channel_layout": "stereo"}},
        "silence": {"duration": 16.0, "threshold_db": float((req.get("parameters") or {}).get("threshold_db", -40.0)), "min_duration": 0.5,
                    "segments": [{"start": 0.0, "end": 3.0, "duration": 3.0, "type": "leading", "runs_to_end": False}], "segment_count": 1},
        "loudness": {"integrated_lufs": -11.0, "loudness_range": 6.0, "true_peak_dbtp": -5.0},
        "duration": {"container_duration": 16.0, "longest_stream_duration": 16.0},
        "integrity": {"status": "PASS", "decode_errors": 0},
        "scene_detection": {"cuts": [{"time": 8.0, "score": 42.0}]},
    }.get(kind, {"ok": True})
    ident = "0" * 60 + "%04x" % (hash(kind) & 0xFFFF)
    return {"id": "obs_" + ident[:16], "asset_id": req["asset_id"], "kind": kind, "data": data, "source": f"{tool}@{CONTRACT['version']}",
            "analysis_id": req.get("analysis_id") or f"analysis-{ident[:16]}", "observed_at": "2026-09-04T10:00:00Z",
            "analysis": {"identity": ident, "analyzer": tool, "analyzer_version": CONTRACT["version"], "parameters": req.get("parameters") or {}, "seconds": 0.01},
            "asset": {"path": req["input"], "fingerprint": "f" * 64, "size": 1000}}


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["contract"]:
        log("contract")
        c = dict(CONTRACT)
        if MODE == "wrong_version":
            c["version"] = "9.0.0"
        if MODE == "wrong_schema_contract":
            c["schema"] = "media-analysis/contract@2"
        print(json.dumps(c))
        return 0
    if argv[:1] == ["doctor"]:
        log("doctor")
        print(json.dumps({"schema": "media-analysis/doctor@1", "skill": {"id": "media-analysis", "version": CONTRACT["version"]}, "status": "ok", "checks": {"ffprobe": {"status": "ok"}}}))
        return 0
    if argv[:1] == ["run"]:
        log("run")
        req = json.loads(sys.stdin.read())
        if MODE == "hang":
            time.sleep(30)
        if MODE == "crash":
            print("boom", file=sys.stderr)
            return 8
        if MODE == "empty":
            return 0
        if MODE == "text":
            print("not json at all")
            return 0
        obs = observation(req)
        cache = {"status": "hit" if os.environ.get("FAKE_MA_CACHE") == "hit" else "miss", "policy": req.get("cache_policy", "use"), "key": "k" * 64}
        result = {"analysis_id": obs["analysis_id"], "asset_id": req["asset_id"], "kind": req["kind"], "status": "ok", "observation": obs, "cache": cache,
                  "usage": {"analyzer_calls": 0 if cache["status"] == "hit" else 1, "seconds": 0.01, "operations": [{"executable": "ffprobe", "purpose": "metadata"}]}}
        if MODE == "wrong_kind":
            result["kind"] = "timing"
        if MODE == "no_observation":
            result.pop("observation")
        if MODE == "bad_source":
            result["observation"]["source"] = "ai:model"
        if MODE == "error_result":
            result = {"analysis_id": obs["analysis_id"], "asset_id": req["asset_id"], "kind": req["kind"], "status": "error",
                      "error": {"code": "ANALYZER_UNAVAILABLE", "message": "ffmpeg missing", "details": {}}, "error_kind": "ANALYZER_UNAVAILABLE",
                      "cache": cache, "usage": {"analyzer_calls": 0, "seconds": 0.0, "operations": []}}
        doc = {"schema": "media-analysis/response@1", "skill": {"id": "media-analysis", "version": CONTRACT["version"]}, "status": result["status"], "dry_run": "--dry-run" in argv,
               "results": [result], "observations": [obs] if result["status"] == "ok" else [], "usage": {"analyzer_calls": 1, "cache_hits": 0, "seconds": 0.01},
               "budget": {"calls": 1, "seconds": 0.01, "budget": {"max_analysis_calls": None, "timeout": 600.0, "max_total_seconds": None}}, "warnings": []}
        if MODE == "wrong_schema":
            doc["schema"] = "media-analysis/response@2"
        if MODE == "wrong_skill":
            doc["skill"]["id"] = "other"
        print(json.dumps(doc))
        if MODE == "two_docs":
            print(json.dumps(doc))
        return 0 if result["status"] == "ok" else 6
    print("unknown", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
