"""Artifact lifecycle (ADR-022): registration, integrity, QA association, delivery promotion and archive.

    ProductionPlan.outputs → (IR → compiler → execution) → Artifact → QA → Delivery → Archive

An Artifact is a production result, not a file: its identity is (project, plan, logical name, sha256), it links the
producing job(s), operations, production step and decisions, and it carries a QA status and a lifecycle stage.
Manifests live in <workspace>/artifacts/registry/<artifact id>.json; the archive index in <workspace>/archive/.
Nothing here writes media, moves files or uploads anywhere: delivery is a recorded promotion, archive a recorded state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..media.analyzer import sha256_file
from ..models import ARTIFACT_STAGES, Artifact, now_iso, stable_hash

DELIVERY_STATUS = {"working": "NOT_READY", "candidate": "READY", "approved": "READY", "final": "DELIVERED", "archive": "ARCHIVED"}
PROMOTIONS = {"working": (), "candidate": ("final", "archive"), "approved": ("final", "archive"), "final": ("archive",), "archive": ()}
FAILURE_KINDS = ("ARTIFACT_MISSING", "ARTIFACT_UNREADABLE", "ARTIFACT_HASH_MISMATCH", "ARTIFACT_OUTSIDE_WORKSPACE", "ARTIFACT_CONFLICT",
                 "ARTIFACT_NOT_DELIVERABLE", "ARTIFACT_REGISTRATION_FAILED")


class ArtifactError(Exception):
    def __init__(self, kind: str, message: str = ""):
        if kind not in FAILURE_KINDS:
            kind = "ARTIFACT_REGISTRATION_FAILED"
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind


def artifact_id(project_id: str, plan_id: str, logical_name: str, sha256: str) -> str:
    """Deterministic content identity: same project + plan + logical output + bytes → same artifact. A revised plan or a
    different file content is a different artifact. Never derived from the path, the job id or a timestamp."""
    return "art_" + stable_hash([project_id, plan_id, logical_name, sha256])[:16]


def delivery_status(stage: str) -> str:
    return DELIVERY_STATUS.get(stage, "NOT_READY")


class ArtifactStore:
    def __init__(self, workspace: str):
        self.workspace = str(Path(workspace).resolve())
        self.registry = Path(self.workspace) / "artifacts" / "registry"
        self.archive_dir = Path(self.workspace) / "archive"

    # ---- security: the artifact must be a regular file inside the workspace (no traversal, no symlink escape)
    def check_path(self, path: str) -> str:
        p = Path(path)
        if not p.is_absolute():
            raise ArtifactError("ARTIFACT_OUTSIDE_WORKSPACE", f"artifact path must be absolute: {path}")
        if any(part == ".." for part in p.parts):
            raise ArtifactError("ARTIFACT_OUTSIDE_WORKSPACE", f"path traversal in artifact path: {path}")
        rp = os.path.normcase(str(p.resolve()))
        ws = os.path.normcase(self.workspace)
        if not rp.startswith(ws + os.sep):
            raise ArtifactError("ARTIFACT_OUTSIDE_WORKSPACE", f"artifact outside workspace: {path}")
        if p.is_symlink():
            raise ArtifactError("ARTIFACT_OUTSIDE_WORKSPACE", f"artifact path is a symlink: {path}")
        return str(p)

    # ---- integrity
    @staticmethod
    def integrity(path: str, expected_sha256: Optional[str] = None, expected_size: Optional[int] = None) -> Dict[str, Any]:
        """exists / readable / size / sha256, and whether they match what the manifest recorded."""
        r: Dict[str, Any] = {"path": path, "exists": os.path.isfile(path), "readable": False, "size": None, "sha256": None, "ok": False, "error": None}
        if not r["exists"]:
            r["error"] = "ARTIFACT_MISSING"
            return r
        try:
            r["size"] = os.path.getsize(path)
            r["sha256"] = sha256_file(path)
            r["readable"] = True
        except OSError as e:
            r["error"] = f"ARTIFACT_UNREADABLE: {e}"
            return r
        if expected_sha256 and r["sha256"] != expected_sha256:
            r["error"] = "ARTIFACT_HASH_MISMATCH"
            return r
        if expected_size is not None and r["size"] != expected_size:
            r["error"] = "ARTIFACT_HASH_MISMATCH"
            return r
        r["ok"] = True
        return r

    # ---- registration
    def register(self, art: Artifact) -> Artifact:
        """Record an artifact manifest. The same identity registered twice must describe the same bytes (immutable);
        a second job that reused the same output is appended to `jobs`, never a second artifact."""
        self.check_path(art.path)
        chk = self.integrity(art.path, art.hash, art.size)
        if not chk["ok"]:
            raise ArtifactError(chk["error"].split(":")[0], f"{art.logical_name}: {chk['error']}")
        self.registry.mkdir(parents=True, exist_ok=True)
        existing = self.get(art.id)
        if existing is not None:
            if existing.hash != art.hash:
                raise ArtifactError("ARTIFACT_CONFLICT", f"{art.id} already registered with different content")
            for j in art.jobs:
                if j not in existing.jobs:
                    existing.jobs.append(j)
            if existing.path != art.path and not self.integrity(existing.path, existing.hash, existing.size)["ok"]:
                # the earlier file is gone or changed: the same bytes now live in the new job's output
                existing.delivery_history.append({"at": now_iso(), "event": "relocated", "from": existing.path, "to": art.path, "job_id": art.job_id})
                existing.path, existing.size = art.path, art.size
            existing.delivery_history.append({"at": now_iso(), "event": "reused", "job_id": art.job_id})
            art = existing
        art.delivery_status = delivery_status(art.stage)
        self._write(art)
        return art

    def _write(self, art: Artifact) -> None:
        p = self.registry / f"{art.id}.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(art.to_dict(), indent=1, ensure_ascii=False, default=str, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)

    def get(self, art_id: str) -> Optional[Artifact]:
        p = self.registry / f"{art_id}.json"
        if not p.exists():
            return None
        return Artifact.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self, project_id: Optional[str] = None, job_id: Optional[str] = None, plan_id: Optional[str] = None) -> List[Artifact]:
        out = []
        for p in sorted(self.registry.glob("*.json")) if self.registry.exists() else []:
            a = Artifact.from_dict(json.loads(p.read_text(encoding="utf-8")))
            if project_id and a.project_id != project_id:
                continue
            if job_id and job_id not in a.jobs:
                continue
            if plan_id and a.plan_id != plan_id:
                continue
            out.append(a)
        return sorted(out, key=lambda a: (a.project_id, a.plan_version, a.logical_name, a.id))

    def verify(self, art_id: str) -> Dict[str, Any]:
        a = self.get(art_id)
        if a is None:
            raise ArtifactError("ARTIFACT_MISSING", art_id)
        return self.integrity(a.path, a.hash, a.size)

    # ---- lifecycle: working → candidate (QA not FAIL) → final (delivered) → archive
    def promote(self, art_id: str, to: str, who: str, reason: str = "", plan_status: Optional[str] = None, channel: str = "local") -> Artifact:
        """Delivery promotion. Gates: integrity (bytes unchanged), QA not FAIL / not PENDING, plan not REJECTED / BLOCKED,
        legal stage transition. `channel` names the delivery target ("local" today; external adapters are future work)."""
        a = self.get(art_id)
        if a is None:
            raise ArtifactError("ARTIFACT_MISSING", art_id)
        if to not in ARTIFACT_STAGES:
            raise ArtifactError("ARTIFACT_NOT_DELIVERABLE", f"unknown stage {to!r}")
        if to not in PROMOTIONS.get(a.stage, ()):
            raise ArtifactError("ARTIFACT_NOT_DELIVERABLE", f"cannot promote {a.id} from {a.stage} to {to}")
        chk = self.integrity(a.path, a.hash, a.size)
        if not chk["ok"]:
            raise ArtifactError(chk["error"].split(":")[0], f"{a.id}: content no longer matches the manifest; a changed file is a new artifact")
        if to == "final":
            if a.qa_status == "FAIL" or a.qa_status in ("PENDING", "UNKNOWN"):
                raise ArtifactError("ARTIFACT_NOT_DELIVERABLE", f"{a.id}: QA {a.qa_status} — only PASS / WARN artifacts can be delivered")
            if plan_status in ("REJECTED", "BLOCKED", "REVIEW", "DRAFT"):
                raise ArtifactError("ARTIFACT_NOT_DELIVERABLE", f"{a.id}: production plan is {plan_status}")
        a.delivery_history.append({"at": now_iso(), "event": "promoted", "from": a.stage, "to": to, "by": who, "reason": reason, "channel": channel})
        a.stage = to
        a.delivery_status = delivery_status(to)
        self._write(a)
        if to == "archive":
            self._archive_index(a)
        return a

    def _archive_index(self, a: Artifact) -> None:
        """Logical archive: per-project index of artifacts with their plan / job / QA / provenance references (no copies, no compression)."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        p = self.archive_dir / f"{a.project_id}.json"
        idx = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"project_id": a.project_id, "entries": []}
        if not any(e["artifact_id"] == a.id for e in idx["entries"]):
            idx["entries"].append({"artifact_id": a.id, "logical_name": a.logical_name, "plan_id": a.plan_id, "plan_version": a.plan_version, "jobs": list(a.jobs),
                                   "sha256": a.hash, "size": a.size, "format": a.format, "role": a.type, "qa_status": a.qa_status, "path": a.path,
                                   "provenance": a.provenance, "archived_at": now_iso()})
        p.write_text(json.dumps(idx, indent=1, ensure_ascii=False, default=str, sort_keys=True), encoding="utf-8")

    def archive_index(self, project_id: str) -> Dict[str, Any]:
        p = self.archive_dir / f"{project_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"project_id": project_id, "entries": []}
