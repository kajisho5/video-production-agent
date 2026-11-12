"""Split tests/test_integration.py across CI runners, one test class at a time.

    python -m tests.integration_shards --shard 2 --of 6     run shard 2 of 6
    python -m tests.integration_shards --of 6 --check       print the split and verify it, run nothing

The split is computed, not listed: every TestCase class the unittest loader finds in tests.test_integration is put in
exactly one shard, so a new class runs without anyone editing this file or the workflow. A class is never divided: its
methods share setUpClass media and a workspace. SECONDS only balances the shards; a class missing from it gets
DEFAULT_SECONDS and still runs.

Before anything runs, the split is verified against the loader's own list of test ids: every id in exactly one shard.
The chosen shard then runs through `unittest.main`, exactly as `python -m unittest -v <class> <class> ...` would.
"""
from __future__ import annotations

import argparse
import sys
import unittest

MODULE = "tests.test_integration"

# Measured seconds per class on the slower engine leg (ffmpeg-skill v2.24.1, ubuntu-latest). Balance only.
# VideoEditingFeaturesRealTests: 222 measured, plus about 64 estimated for ADR-112's end-to-end stinger case (its local time at the class's CI / local ratio).
SECONDS = {
    "RealMediaTests": 758, "IntegratedPipelineRealTests": 710, "AudioProductionRealTests": 309,
    "VideoEditingFeaturesRealTests": 286, "VideoEditingOperationsRealTests": 192, "MediaAnalysisRealTests": 191,
    "SubtitleFeaturesRealTests": 127, "VoiceEnhanceRealTests": 108, "SubtitleMuxRealTests": 96,
    "ThumbnailFeaturesRealTests": 83, "MotionGraphicsFeaturesRealTests": 82, "NleBridgeRealTests": 69,
    "PriorityAToolsRealTests": 68, "VideoEditingRealTests": 66, "DocumentVerifyRealTests": 57,
    "IngestOffloadRealTests": 50, "CropProposalRealTests": 48, "MultiSourceSyncRealTests": 46,
    "NoAudioSourceTests": 45, "FfmpegSkillContractTests": 44, "CameraSwitchRealTests": 44,
    "TakeCrossfadeRealTests": 42, "ImagePrepareRealTests": 39, "SlideSyncRealTests": 36,
    "FaceRedactionRealTests": 34, "TimelineExportRealTests": 33, "DubbingRealTests": 32, "SpeakerIdsRealTests": 32,
    "SoundDesignBgmRealTests": 30, "ThumbnailDocumentRealTests": 23, "IngestFetchRealTests": 22,
    "DiarizationRealTests": 22, "RemotionRealTests": 22, "BeatGridRealTests": 18, "TranscriptionRealTests": 18,
    "PublishRealTests": 2, "ProductionContextRealTests": 1, "ProductionDecisionEngineRealTests": 1,
    "SpeechToPlanRealTests": 1,
}
DEFAULT_SECONDS = 60


def discover() -> dict:
    """{class name: [test id, ...]} for every test the loader finds in the module."""
    classes: dict = {}
    stack = [unittest.defaultTestLoader.loadTestsFromName(MODULE)]
    while stack:
        item = stack.pop()
        if isinstance(item, unittest.TestSuite):
            stack.extend(item)
            continue
        if type(item).__module__ != MODULE:   # unittest's stand-in for a module that failed to import
            raise SystemExit(f"could not load {MODULE}: {item.id()}")
        classes.setdefault(type(item).__name__, []).append(item.id())
    if not classes:
        raise SystemExit(f"no tests found in {MODULE}")
    return {name: sorted(ids) for name, ids in sorted(classes.items())}


def split(names, shards: int) -> list:
    """Heaviest class first into the lightest shard so far; shards returned lightest first (shard 1 is the lightest)."""
    weight = lambda n: SECONDS.get(n, DEFAULT_SECONDS)   # noqa: E731
    bins = [[] for _ in range(shards)]
    for name in sorted(names, key=lambda n: (-weight(n), n)):
        min(bins, key=lambda b: sum(map(weight, b))).append(name)
    bins.sort(key=lambda b: (sum(map(weight, b)), b))
    return [sorted(b) for b in bins]


def verify(classes: dict, bins: list) -> None:
    """Every discovered test id is in exactly one shard, and no shard is empty."""
    wanted = sorted(i for ids in classes.values() for i in ids)
    got = sorted(i for b in bins for name in b for i in classes[name])
    problems = []
    if got != wanted:
        problems.append(f"shards cover {len(got)} test ids ({len(set(got))} distinct), the module has {len(wanted)}")
    for name in classes:
        holders = [n for n, b in enumerate(bins, 1) if name in b]
        if len(holders) != 1:
            problems.append(f"{name} is in shards {holders}, expected exactly one")
    problems += [f"shard {n} is empty" for n, b in enumerate(bins, 1) if not b]
    if problems:
        raise SystemExit("integration shard check failed:\n  " + "\n  ".join(problems))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--of", type=int, required=True, help="number of shards")
    ap.add_argument("--shard", type=int, help="shard to run, 1..OF")
    ap.add_argument("--check", action="store_true", help="print and verify the split, run nothing")
    args = ap.parse_args(argv)
    if args.of < 1 or (not args.check and not (args.shard and 1 <= args.shard <= args.of)):
        ap.error("--shard must be within 1..--of (or pass --check)")
    classes = discover()
    bins = split(classes, args.of)
    verify(classes, bins)
    for n, b in enumerate(bins, 1):
        tests = sum(len(classes[c]) for c in b)
        secs = sum(SECONDS.get(c, DEFAULT_SECONDS) for c in b)
        print(f"shard {n}/{args.of}: {len(b)} classes, {tests} tests, ~{secs}s: {' '.join(b)}")
    unknown = sorted(set(classes) - set(SECONDS))
    if unknown:
        print(f"no measured duration (balanced as {DEFAULT_SECONDS}s each): {' '.join(unknown)}")
    print(f"total: {len(classes)} classes, {sum(map(len, classes.values()))} tests, each in exactly one shard", flush=True)
    if args.check:
        return
    unittest.main(module=None, argv=["unittest", "-v"] + [f"{MODULE}.{c}" for c in bins[args.shard - 1]])


if __name__ == "__main__":
    main()
