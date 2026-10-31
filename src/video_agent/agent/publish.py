"""Publishing a QC-passed deliverable to a platform (ADR-084; publish-skill roadmap step 6, agent side: SkillSpec publish_upload).

    Requirement (`--set publish.platform=<id>` + metadata, explicit only) → Decision `publish.upload` (DELIVER, always CONFIRM: outward-facing
    and irreversible) → step `publish_upload` after the QC gate → IR `delivery.publish` → render: publish-skill's offline `plan` and the dry
    run of `upload` on the registered deliverable (nothing uploaded, a confirm_token) → `video-agent publish <artifact> --confirm <token>`,
    typed by a human, after `deliver`: the one upload, sent once, never retried.

Vocabulary, decisions and checks only, no execution (Service.publish runs the upload):

- **Requirements** (`publish.` keys; never from a profile, a brief or a default): `publish.platform` (required; one of the platforms the
  pinned publish-skill contract implements with an uploader -- read from it, never listed here), `publish.title`, `publish.description`,
  `publish.tags` (comma-separated), `publish.privacy` (the platform's own `privacy.values`; default private, so a public-only platform
  needs an explicit `public`), `publish.thumbnail=true` / `publish.captions=true` (attach this plan's own thumbnail / subtitle sidecar),
  `publish.target` (which delivery target, when the profile has several). A field the platform does not take is refused here, as the
  Skill would (METADATA_INVALID).
- **Rules the plan checks**: the QC gate is on (`qc=true`: publishing comes after QC, publish-skill's first rule); one deliverable; a
  picture (not the audio production path); a planned thumbnail / sidecar when asked to attach one; the delivery target exists. Each is a
  BLOCK with the reason. Without publish-skill the step has no tool and the plan's handoff names the package (ADR-051).
- **The approval** is the existing gate: the decision is CONFIRM with a CONFIRM floor (an explicit request does not waive it), so render
  waits for `approve`. Approving the plan is not approving the bytes: render stops at the dry run, and the upload needs the token of that
  dry run, typed by a human, for the artifact they delivered."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from ..tools.publish.adapter import LANG_RE, MAX_TEXT, platforms as contract_platforms, privacy_values
from .audio import SWITCH as AUDIO_SWITCH
from .editing import EditRequirementError, _bool

PREFIX = "publish."
PACKAGE = "publish"
SUBJECT = "publish.upload"
SKILL = "publish_upload"
REQUIREMENT_KEYS = ("publish.platform", "publish.title", "publish.description", "publish.tags", "publish.privacy", "publish.thumbnail", "publish.captions", "publish.target")
DEFAULT_PRIVACY = "private"     # publish-skill's own default (metadata.DEFAULT_PRIVACY); a platform that has no private posts needs `public` named
APPROVAL_KEYS = {SUBJECT: (f"{SUBJECT}.approval", "CONFIRM")}
MAX_TAGS = 500


def _text(v: Any, key: str) -> str:
    if not isinstance(v, str) or not v.strip() or "\x00" in v or len(v) > MAX_TEXT:
        raise EditRequirementError(f"{key} must be non-empty text of at most {MAX_TEXT} characters")
    return v


def parse_publish_requirements(m: Dict[str, Requirement]) -> Optional[Dict[str, Any]]:
    """Requirement map → {"platform", "meta": {title, description, tags, privacy}, "attach": {thumbnail, captions}, "target", "requirements"} or
    None when nothing is asked. Only USER requirements count (publishing is opt-in per request)."""
    given = {k: r for k, r in m.items() if k.startswith(PREFIX)}
    if not given:
        return None
    unknown = sorted(k for k in given if k not in REQUIREMENT_KEYS)
    if unknown:
        raise EditRequirementError(f"unknown requirement key(s): {', '.join(unknown)}; publish.* keys are {', '.join(REQUIREMENT_KEYS)}")
    not_user = sorted(k for k, r in given.items() if r.provenance != "USER")
    if not_user:
        raise EditRequirementError(f"{', '.join(not_user)}: publishing is opt-in on the request (--set), never from a profile, a brief or a default")
    if "publish.platform" not in given:
        raise EditRequirementError(f"{sorted(given)[0]} needs publish.platform (one of {', '.join(sorted(contract_platforms()))})")
    plats = contract_platforms()
    pid = given["publish.platform"].value
    if pid not in plats:
        raise EditRequirementError(f"publish.platform {pid!r} has no uploader in publish-skill's contract (implemented: {', '.join(sorted(plats))})")
    fields = set(plats[pid]["metadata_fields"])
    meta: Dict[str, Any] = {}
    for k in ("title", "description"):
        if f"publish.{k}" in given:
            meta[k] = _text(given[f"publish.{k}"].value, f"publish.{k}")
    if "publish.tags" in given:
        v = given["publish.tags"].value
        tags = [t.strip() for t in (v if isinstance(v, list) else str(v).split(",")) if str(t).strip()]
        if not tags or len(tags) > MAX_TAGS:
            raise EditRequirementError(f"publish.tags must name 1..{MAX_TAGS} tags (comma-separated)")
        meta["tags"] = [_text(t, "publish.tags") for t in tags]
    privacy = given["publish.privacy"].value if "publish.privacy" in given else DEFAULT_PRIVACY
    allowed = plats[pid]["privacy"] or list(privacy_values())
    if privacy not in allowed:
        raise EditRequirementError(f"publish.privacy {privacy!r} is not one {pid} takes ({', '.join(allowed)})"
                                   + ("; a post there is public the moment it is made: say publish.privacy=public only if that is meant" if allowed == ["public"] else ""))
    meta["privacy"] = privacy
    attach = {k: _bool(given[f"publish.{k}"].value, f"publish.{k}") for k in ("thumbnail", "captions") if f"publish.{k}" in given}
    for k in [k for k in meta if k != "privacy"] + [k for k, on in attach.items() if on]:
        if k not in fields:
            raise EditRequirementError(f"publish.{k}: {pid} does not take {k} through publish-skill (it takes {', '.join(sorted(fields))})")
    target = given["publish.target"].value if "publish.target" in given else None
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise EditRequirementError("publish.target must name a delivery target id")
    return {"platform": pid, "meta": meta, "attach": attach, "target": target, "requirements": [given[k] for k in REQUIREMENT_KEYS if k in given]}


def check_publish_keys(user_requirements: Dict[str, Any]) -> None:
    """The `--set` gate for `publish.*` (before any analysis runs): known keys, an implemented platform, fields and privacy it takes."""
    if not any(k.startswith(PREFIX) for k in user_requirements or {}):
        return
    parse_publish_requirements({k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in user_requirements.items() if k.startswith(PREFIX)})


def decide_publish(eng: Any, m: Dict[str, Requirement], decs: List[Any], subjects: List[Tuple[str, List[str]]], targets: List[Dict[str, Any]],
                   approval_for: Callable[..., Dict[str, Any]], probe_ids_of: Callable[[List[str]], List[str]], cap_block: Callable[[str, str], Any]) -> None:
    """One `publish.upload` decision: DELIVER with a CONFIRM floor, or BLOCK with the reason."""
    p = parse_publish_requirements(m)
    if p is None:
        return
    req = p["requirements"]
    subject = subjects[0][0] if subjects else "programme"
    sources = [s for _, srcs in subjects for s in srcs]
    ev = [r.id for r in req] + probe_ids_of(sources)

    def block(why: str) -> None:
        eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: publish to {p['platform']}", reason=why + "; nothing is uploaded", confidence=1.0, evidence=ev, risk="HIGH",
                   approval="BLOCK", provenance="USER", params={"asset_id": subject, "platform": p["platform"]}, requirements=req)

    sw = m.get(AUDIO_SWITCH)
    if sw is not None and sw.provenance != "DEFAULT" and _bool(sw.value, AUDIO_SWITCH):
        return block("the audio production path delivers audio; publish-skill uploads a video deliverable")
    if len(subjects) != 1:
        return block(f"{len(subjects)} separate deliverables: one publish names one deliverable (join the inputs into one programme, or publish them one per request)")
    live = [d for d in decs if d.status != "REJECTED"]
    qc = next((d for d in live if d.subject == "qc.check" and d.type == "DELIVER" and d.params.get("asset_id") == subject), None)
    if qc is None:
        return block("publishing comes after QC (publish-skill's first rule): turn the QC gate on with --set qc=true")
    if p["attach"].get("thumbnail") and not any(d.subject == "thumbnail.render" and d.type == "TRANSFORM" and d.params.get("asset_id") == subject for d in live):
        return block("publish.thumbnail=true attaches this plan's thumbnail, and none is planned (--set thumbnail=true)")
    if p["attach"].get("captions") and not any(d.subject == "subtitle.generate" and d.type == "TRANSFORM" and d.params.get("asset_id") == subject for d in live):
        return block("publish.captions=true attaches this plan's subtitle sidecar, and none is planned (--set subtitle=true)")
    ids = [t.get("id") for t in targets]
    if p["target"] is not None:
        target = p["target"] if p["target"] in ids else None
        if target is None:
            return block(f"publish.target {p['target']!r} is not a delivery target of this plan ({', '.join(str(i) for i in ids) or 'none'})")
    else:
        same = [t.get("id") for t in targets if t.get("platform") == p["platform"]]
        target = same[0] if len(same) == 1 else (ids[0] if len(ids) == 1 else None)
        if target is None:
            return block(f"{len(ids)} delivery targets and none for {p['platform']}: name the one to publish with publish.target")
    caption_lang = next((d.params.get("language") for d in live if d.subject == "subtitle.generate" and d.params.get("asset_id") == subject), None) if p["attach"].get("captions") else None
    if p["attach"].get("captions") and not LANG_RE.match(str(caption_lang or "")):
        return block(f"the subtitle sidecar's language {caption_lang!r} is not a BCP-47 tag the platform's captions need")
    shown = ", ".join(f"{k} {v!r}" if k != "tags" else f"{len(v)} tag(s)" for k, v in p["meta"].items())
    attached = [k for k, on in p["attach"].items() if on]
    eng.decide(subject=SUBJECT, type="DELIVER", decision=f"publish the {target} deliverable of {subject} to {p['platform']} ({shown}" + (f"; with its {' and '.join(attached)}" if attached else "") + ")",
               reason=f"user asked to publish ({req[0].source}); publishing is outward-facing and cannot be undone, so it always needs confirmation; it runs only after the "
                      "QC gate passed: render checks the deliverable and metadata against the platform's limits offline (publish-skill plan) and makes the dry run "
                      "(nothing uploaded); the upload itself needs the dry run's confirmation token, typed by a human with `video-agent publish <artifact> --confirm "
                      "<token>` after `deliver`, and is sent once (never retried); credentials stay in the user's environment or OS keychain",
               confidence=1.0, evidence=ev, risk="HIGH", approval=approval_for(SUBJECT, floor="CONFIRM", keys=APPROVAL_KEYS), provenance="USER",
               params={"asset_id": subject, "platform": p["platform"], "target": target, **p["meta"], "thumbnail": bool(p["attach"].get("thumbnail")),
                       "captions": bool(p["attach"].get("captions")), "caption_language": caption_lang, "after_qc": qc.id, "lifecycle": "EXPERIMENTAL"},
               requirements=req, serves_intent=None)
    cap_block(SKILL, f"capability.{SKILL}")
