"""MCP server (ADR-050): the video-agent CLI as tools for a chat app -- Claude Desktop, Claude Code, any MCP client.

Stdio JSON-RPC 2.0, standard library only. Each tool runs `video-agent --json <command> ...` in a child process, so
the protocol stream on stdout never sees a skill's output, and the result is exactly what the CLI prints. The chat
model is the reasoning step (providers/host.py): `plan` returns an ai_brief, the model answers it, and `revise` takes
that answer as `ai_recommendations`, validated like any provider's. A brief (ADR-061) works the same way: `brief` (or `plan`
with `brief`) returns `host_request`, the model answers it, and `plan` takes the answer as `brief_requirements`, validated like
`--set`. Approval and rendering stay separate tools: the
server never approves on its own, and `render` has no inline approve.

The child is started without the current directory on its import path (ADR-075): the host's working directory may be any
folder, and a `json.py` or `video_agent/` there must not run as the agent.

Permission scopes (ADR-125, mcp_permissions.py): every tool call has a class (read / write / destructive / publish) and runs only
when that class is enabled (default read + write); a disabled call returns a structured error naming the setting that enables it.
A destructive or publish call returns a preview and a single-use, short-lived confirm_token first, and runs only when repeated with it.

    video-agent-mcp [--workspace DIR] [--ffmpeg-skill-dir DIR] [--scopes read,write,...] [--allow-root DIR] [--config FILE]
                    [--confirm-ttl S]      (or python -m video_agent.mcp_server, see ADR-075)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import __version__
from .mcp_permissions import (CONFIRMED_CLASSES, DESTRUCTIVE, ENV_AI_PROVIDER, ESCALATES, PUBLISH, READ, TOOL_CLASSES, ConfirmationTokens, Permissions,
                              PermissionConfigError, call_digest, classify, enable_hint, load_permissions)
from .tools.discovery import python_module_command, safe_command

PROTOCOL_VERSION = "2024-11-05"
TOOL_TIMEOUT_S = 6 * 3600   # a render of a long recording can take hours; the client may cancel sooner

_STR = {"type": "string"}
_PROJECT = {"type": "string", "description": "path of the project.json that plan printed"}
_ENT_PROJECT = {"type": "string", "description": "a project's entity store (an event, a series); default the global one"}
_ENT_TYPE = {"type": "string", "enum": ["person", "organization", "product", "place", "term"]}
_ENT_FIELDS = {"canonical": {"type": "string", "description": "the spelling to use"}, "type": _ENT_TYPE,
               "aliases": {"type": "array", "items": _STR, "description": "other spellings and common mis-hearings"}, "title": {"type": "string", "description": "role, for a lower third"},
               "pronunciation": {"type": "array", "items": {"type": "object", "required": ["lang", "reading"], "properties": {"lang": _STR, "reading": _STR, "ipa": _STR, "accent": {"type": "integer"}}}},
               "speaker_id": {"type": "string", "description": "diarization-skill's anonymous id (SPEAKER_00, ...)"},
               "voiceprint": {"type": "string", "description": "a diarization-skill/voiceprint@1 file of the person's enrolled voice"}, "note": _STR}

_RENAME_PROPS = {"folder": {"type": "string", "description": "the offloaded copy (with its ingest-manifest.json)"},
                 "template": {"type": "string", "description": "ingest-skill template, default {date}_{event}_{room}_{camera}_{scene:S02}_{speaker:Speaker-}_{seq:04}{ext}"},
                 "event": _STR, "room": _STR, "camera": _STR, "set": {"type": "array", "items": _STR, "description": "more fields, FIELD=VALUE"},
                 "evidence": {"type": "array", "items": _STR, "description": "folders / files of diarization turns or name proposals and transcripts of the clips"},
                 "entity_project": _STR, "no_entities": {"type": "boolean"}, "scene_gap": {"type": "number"}, "no_scenes": {"type": "boolean"},
                 "utc_offset": _STR, "date_from_mtime": {"type": "boolean"},
                 "copy_to": {"type": "string", "description": "renamed verified copies here instead of renaming in place"}}

TOOLS: List[Dict[str, Any]] = [
    {"name": "doctor", "description": "Inspect the environment: ffmpeg, the production skills and their status (AVAILABLE / MISSING / DEGRADED).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "skills", "description": "List the production skills with their status here and the selected tool.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "plan", "description": (
        "Analyze media files and write a production plan (Project IR). With ai_brief (default true) the result also has "
        "`ai_brief`: read it, decide recommendations from the cited facts only, then call `revise` with them. "
        "Show the user the decisions; nothing is rendered here."),
     "inputSchema": {"type": "object", "required": ["inputs"], "properties": {
         "inputs": {"type": "array", "items": _STR, "description": "local media file paths (or URLs)"},
         "profile": {"type": "string", "description": "delivery profile, e.g. youtube, broadcast, generic (default generic)"},
         "request": {"type": "string", "description": "the user's request in their own words"},
         "set": {"type": "object", "description": "explicit user requirements, e.g. {\"audio.loudness.target_lufs\": -16}"},
         "name": _STR, "language": {"type": "string", "description": "ISO 639-1 language for a transcript"},
         "kinds": {"type": "array", "items": _STR, "description": "extra analysis kinds, e.g. transcript, scene_detection, crop_rectangle (black bars → a CONFIRM crop proposal), speaker_turns (diarization-skill)"},
         "ai_brief": {"type": "boolean", "description": "return the brief for the chat model (default true)"},
         "ai_model": {"type": "string", "description": "name of the chat model, recorded in provenance"},
         "brief": {"type": "string", "description": "the user's brief in their own words, read into requirement keys (explicit `set` wins)"},
         "brief_requirements": {"type": ["object", "array"], "description": "your answer to the brief's host_request: {\"requirements\": [{key, value, phrase, confidence}], \"unmapped\": [...]}"},
         "channel": {"type": "string", "description": "the channel / brand / series this video belongs to: its stored preferences apply (ADR-111)"},
         "no_preferences": {"type": "boolean", "description": "plan without the stored preferences"},
         "entity_project": {"type": "string", "description": "the entity memory's project store applied on top of the global one (default: channel)"},
         "no_entities": {"type": "boolean", "description": "plan without the entity memory"}}}},
    {"name": "brief", "description": (
        "Read the user's brief (e.g. \"60 s vertical, captions, -14 LUFS\") into the agent's requirement keys without analysing media. "
        "Without brief_requirements the result has `host_request`: answer it with keys from its allowed_keys only, each quoting the "
        "brief's words, then call `brief` or `plan` again with brief_requirements. Every key is checked like --set; unknown keys and "
        "invalid values are refused with the reason. Without an answer the rule-based reading is shown."),
     "inputSchema": {"type": "object", "required": ["text"], "properties": {
         "text": {"type": "string", "description": "the brief"}, "set": {"type": "object", "description": "explicit requirements (they win)"},
         "brief_requirements": {"type": ["object", "array"], "description": "your answer to host_request"},
         "ai_model": {"type": "string", "description": "name of the chat model that answered"}}}},
    {"name": "revise", "description": (
        "Write the next plan version. Pass the chat model's answer to the ai_brief as ai_recommendations "
        "({\"recommendations\": [{intent, asset_id, statement, confidence, evidence: [ids], params}]}); every one is "
        "validated (cited ids must exist, unknown intents are dropped) and the result lists accepted and dropped ones. "
        "Also takes the user's feedback after a reject."),
     "inputSchema": {"type": "object", "required": ["project"], "properties": {
         "project": _PROJECT, "feedback": {"type": "string"}, "set": {"type": "object"}, "by": _STR,
         "ai_recommendations": {"type": ["object", "array"], "description": "the answer to the ai_brief"},
         "ai_model": {"type": "string", "description": "name of the chat model that answered"},
         "scope": {"type": "string", "enum": ["this_video", "channel", "all_videos"], "description": "what the feedback is about (default this_video). Say a wider scope only when the user said it"},
         "scope_by": {"type": "string", "enum": ["user", "ai"], "description": "ai when you chose the wider scope yourself: it is then only a candidate the user confirms at promotion"},
         "channel": {"type": "string", "description": "name the project's channel (once; needed for scope channel)"}}}},
    {"name": "preferences", "description": (
        "The versioned preference stores (ADR-111). action promote stores one feedback item's --set keys in the channel's store (scope channel) or the "
        "global one (all_videos): call it only after the user asked for this preference to be kept for future videos. list / show / diff / rollback "
        "read or restore versions (a rollback adds a new version)."),
     "inputSchema": {"type": "object", "required": ["action"], "properties": {
         "action": {"type": "string", "enum": ["promote", "list", "show", "diff", "rollback"]}, "project": _PROJECT,
         "feedback": {"type": "string", "description": "promote: the feedback id"}, "confirm_scope": {"type": "string", "enum": ["channel", "all_videos"]},
         "channel": {"type": "string", "description": "the channel's store (default global)"}, "version": {"type": "integer"},
         "from": {"type": "integer"}, "to": {"type": "integer"}, "by": _STR, "reason": _STR}}},
    # ADR-128: the entity memory. Names of real people: they stay in the workspace and reach only the local Skills a plan uses them with
    {"name": "entities_list", "description": "The entity memory (people, organisations, products, places, terms with their spellings, aliases, titles, readings) of the global store or a project's.",
     "inputSchema": {"type": "object", "properties": {"project": _ENT_PROJECT, "type": _ENT_TYPE}}},
    {"name": "entities_add", "description": "Add one entity the user named (never one you guessed: suggest first and let the user confirm).",
     "inputSchema": {"type": "object", "required": ["canonical", "type"], "properties": {"project": _ENT_PROJECT, **_ENT_FIELDS, "id": _STR}}},
    {"name": "entities_edit", "description": "Change an entity (by id, spelling or alias): fields given replace the stored ones; add_aliases / remove_aliases; clear removes fields.",
     "inputSchema": {"type": "object", "required": ["ref"], "properties": {"project": _ENT_PROJECT, "ref": _STR, **{k: v for k, v in _ENT_FIELDS.items() if k != "aliases"},
                                                                       "add_aliases": {"type": "array", "items": _STR}, "remove_aliases": {"type": "array", "items": _STR},
                                                                       "clear": {"type": "array", "items": {"enum": ["aliases", "title", "pronunciation", "speaker", "note"]}}}}},
    {"name": "entities_remove", "description": "Remove an entity (destructive: returns a preview and a confirm_token first).",
     "inputSchema": {"type": "object", "required": ["ref"], "properties": {"project": _ENT_PROJECT, "ref": _STR}}},
    {"name": "entities_import", "description": "Merge a CSV (or an entity store JSON) file into the store by id; replace=true makes the store exactly the file (destructive, confirmed).",
     "inputSchema": {"type": "object", "required": ["file"], "properties": {"project": _ENT_PROJECT, "file": _STR, "replace": {"type": "boolean"}}}},
    {"name": "entities_export", "description": "The store as CSV (default) or JSON: returned, or written to output.",
     "inputSchema": {"type": "object", "properties": {"project": _ENT_PROJECT, "format": {"enum": ["csv", "json"]}, "output": _STR}}},
    {"name": "entities_suggest", "description": "Candidate entities from a plan's transcripts (recurring capitalised names, low-confidence name-like words). Nothing is added: show them to the user, then entities_add the ones they confirm.",
     "inputSchema": {"type": "object", "required": ["plan"], "properties": {"project": _ENT_PROJECT, "plan": _PROJECT, "min_count": {"type": "integer", "minimum": 1}}}},
    # ADR-129: smart rename of an offloaded copy through ingest-skill `rename` (preview reads; apply / undo rename files and are always confirmed)
    {"name": "rename_preview", "description": (
        "Smart rename preview: for an offloaded copy (a folder ingest offload wrote), every clip's new name from the template (date, event, room, camera, "
        "scene, speaker from the entity memory via diarization / transcripts given as evidence, seq) with each field's source. Nothing is renamed. Show the "
        "user the rows; only after their yes call rename_apply with the returned plan_id."),
     "inputSchema": {"type": "object", "required": ["folder"], "properties": _RENAME_PROPS}},
    {"name": "rename_apply", "description": "Rename exactly the previewed plan (plan_id from rename_preview): destructive, returns a preview and a confirm_token first.",
     "inputSchema": {"type": "object", "required": ["folder", "plan_id"], "properties": {**_RENAME_PROPS, "plan_id": _STR}}},
    {"name": "rename_undo", "description": "Restore the original names of the last rename of a folder (or of the given undo manifest): destructive, confirmed.",
     "inputSchema": {"type": "object", "required": ["folder"], "properties": {"folder": _STR, "undo": {"type": "string", "description": "an ingest-rename/<time>.json file"}}}},
    {"name": "corrections", "description": "Corrections per minute of a project per plan version, vs. the previous version and the channel's recent average. A report only.",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {"project": _PROJECT}}},
    {"name": "validate", "description": "Validate a Project IR (schema, semantics, capabilities).",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {"project": _PROJECT}}},
    {"name": "explain", "description": "Why was this decided: reason, evidence, alternatives, AI provenance. Without decision: every decision.",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {
         "project": _PROJECT, "decision": {"type": "string", "description": "decision id or subject"},
         "pipeline": {"type": "boolean", "description": "the whole chain request -> ... -> artifacts"}}}},
    {"name": "diff", "description": "PlanDiff between plan versions (default previous -> current).",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {"project": _PROJECT}}},
    {"name": "approve", "description": "Approve CONFIRM decisions. Call only after the user has said, in this conversation, that they approve these decisions.",
     "inputSchema": {"type": "object", "required": ["project", "decision"], "properties": {
         "project": _PROJECT, "decision": {"type": "string", "description": "comma separated decision ids, or all"},
         "reason": _STR, "by": {"type": "string", "description": "who approved (the user's name)"}}}},
    {"name": "reject", "description": "Reject decisions with the user's reason; then call revise.",
     "inputSchema": {"type": "object", "required": ["project", "decision", "reason"], "properties": {
         "project": _PROJECT, "decision": _STR, "reason": _STR, "by": _STR}}},
    {"name": "preflight", "description": (
        "Check an approved plan's deliverables before rendering, without encoding anything: the output directory and free disk space, the "
        "preset / platform and the platform limits the contracts document, every input still there and unchanged, fonts and sidecars, the "
        "loudness target, the tools. Returns {ok, problems: [{code, severity, message, field}], estimates, unverified}. render runs it too."),
     "inputSchema": {"type": "object", "required": ["project"], "properties": {"project": _PROJECT}}},
    {"name": "render", "description": ("Execute an approved plan through the skills, then QA. dry_run shows the commands without running them. Pending CONFIRM "
                                       "decisions block it: approve them first. The preflight runs first and its errors refuse the render; skip_preflight "
                                       "(recorded in the run) only after the user agreed to render despite them."),
     "inputSchema": {"type": "object", "required": ["project"], "properties": {
         "project": _PROJECT, "dry_run": {"type": "boolean"}, "resume": {"type": "string", "description": "job id or last"},
         "skip_preflight": {"type": "boolean", "description": "render even when the preflight reports errors (recorded in the run)"}}}},
    {"name": "check", "description": "Probe an output file and check platform compliance (loudness, codec, resolution).",
     "inputSchema": {"type": "object", "required": ["output"], "properties": {
         "output": _STR, "platform": {"type": "string", "description": "e.g. youtube (default custom)"}, "qc": {"type": "boolean"}}}},
    {"name": "selects", "description": (
        "Batch selects (preview only): for 1-50 clips (a folder or files), media-analysis-skill and qc-skill measure every shot (cuts, blur, exposure, "
        "camera shake, black / frozen frames, silence, speech) and rule selects@1 keeps or rejects each one with its reasons. Nothing is rendered. "
        "Show the user the returned lines (every shot, why, the total duration) and ask; only after their yes call `plan` with the clips and "
        "set {\"edit.selects\": <path>}, then `approve` the selects.keep decisions."),
     "inputSchema": {"type": "object", "required": ["inputs"], "properties": {
         "inputs": {"type": "array", "items": _STR, "description": "a folder of clips, or the clip files (1-50)"},
         "max_duration": {"type": "number", "description": "keep the best-scored usable shots that fit this many seconds"},
         "min_score": {"type": "number", "description": "reject usable shots scoring below this (0-100)"},
         "max_blur": {"type": "number", "description": "soft above this blurdetect value (default: relative to the batch)"},
         "max_jitter": {"type": "number", "description": "shaky above this camera jitter, % of the width (default 1.0)"},
         "min_shot": {"type": "number", "description": "shortest usable shot in seconds (default 1.0)"},
         "reject_dead_air": {"type": "boolean", "description": "reject silent / speechless shots (default: flagged only)"},
         "output": {"type": "string", "description": "where to write the selects file"}}}},
    {"name": "transcript_edit", "description": (
        "Preview a transcript edit: the user names words (index ranges) or sentences (segment ids) of the transcript to remove, or to keep. "
        "transcription-skill finds their times and gaps, video-editing-skill places each cut on silence and on a frame and retimes the "
        "transcript and subtitles. Nothing is cut. Show the user the returned lines (each cut's seconds, speakers and sentences, and the total) "
        "and ask; only after their yes call `plan` with set {\"edit.transcript_edit\": <path>} and `approve` the transcript.ripple decision. "
        "Never choose the words yourself here."),
     "inputSchema": {"type": "object", "required": ["input", "transcript", "mode"], "properties": {
         "input": {"type": "string", "description": "the video file"}, "transcript": {"type": "string", "description": "its transcript JSON (word timings)"},
         "mode": {"type": "string", "enum": ["remove", "keep"]},
         "words": {"type": "string", "description": "word index ranges, e.g. 12-40,55"}, "segments": {"type": "string", "description": "segment ids, e.g. seg_0003,seg_0007"},
         "include_overlap": {"type": "boolean", "description": "also remove kept words that overlap the removed speech (default: refused)"},
         "subtitles": {"type": "array", "items": _STR, "description": "SRT / VTT files of the input to retime"},
         "output": {"type": "string", "description": "where to write the transcript-edit file"}}}},
    {"name": "artifacts", "description": "Registered artifacts of a project (or all), with QA status.",
     "inputSchema": {"type": "object", "properties": {"project": _PROJECT}}},
]


_CONFIRM_TOKEN = {"type": "string", "description": "the confirm_token a preview of this same call returned; pass it only after the user agreed to "
                  "exactly what the preview lists (single use, expires after a few minutes)"}


def listed_tools(perms: Permissions) -> List[Dict[str, Any]]:
    """TOOLS as tools/list shows them on this server: each with its permission class in the description and as MCP annotations,
    a confirm_token argument on tools whose calls can need confirmation, and a note on what this server's scopes leave disabled."""
    out = []
    for t in TOOLS:
        name = t["name"]
        cls = TOOL_CLASSES.get(name)
        if cls is None:   # fail closed: refused when called; shown as such
            out.append({**t, "description": t["description"] + " [no permission class: refused on this server]"})
            continue
        esc = ESCALATES.get(name, ())
        note = f" [permission class: {cls}" + (f"; a call can be {' / '.join(esc)}, which needs a preview and a confirm_token" if esc else
                                               "; every call needs a preview and a confirm_token" if cls in CONFIRMED_CLASSES else "")
        off = [c for c in (cls, *esc) if not perms.enabled(c)]
        if off:
            note += f"; {' / '.join(off)} disabled on this server"
        schema = dict(t["inputSchema"])
        if esc or cls in CONFIRMED_CLASSES:   # ADR-128: entities_remove is destructive on every call
            schema["properties"] = {**schema.get("properties", {}), "confirm_token": _CONFIRM_TOKEN}
        out.append({**t, "description": t["description"] + note + "]", "inputSchema": schema,
                    "annotations": {"readOnlyHint": cls == READ and not esc, "destructiveHint": DESTRUCTIVE in (cls, *esc),
                                    "openWorldHint": PUBLISH in (cls, *esc)}})
    return out


def _sets(values: Optional[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for k, v in (values or {}).items():
        out.append(f"--set={k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}")
    return out


def _brief_file(a: Dict[str, Any], tmpdir: str) -> List[str]:
    if a.get("brief_requirements") is None:
        return []
    path = os.path.join(tmpdir, "brief_requirements.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(a["brief_requirements"], f, ensure_ascii=False)
    return [f"--brief-requirements={path}"]


def tool_argv(name: str, a: Dict[str, Any], tmpdir: str) -> List[str]:
    """CLI arguments for a tool call (after the global options). Raises ValueError on a missing argument.

    Every value a client sends is data, never a CLI option (ADR-069): options are spelled `--flag=value`, so a value
    beginning with '-' is that option's value, and the positionals (inputs / project / output / text) come last, after a
    `--` separator, so an input such as `--output=/elsewhere.json` is a file name the CLI then fails to find instead of an
    option that redirects where the Project IR is written."""
    def need(key: str) -> str:
        v = a.get(key)
        if v in (None, "", []):
            raise ValueError(f"{name}: '{key}' is required")
        return str(v)

    def opt(flag: str, key: str) -> List[str]:
        return [f"{flag}={a[key]}"] if a.get(key) not in (None, "") else []

    def cmd(sub: str, options: List[str], positionals: List[str]) -> List[str]:
        return [sub, *options, "--", *positionals]

    if name in ("doctor", "skills"):
        return [name]
    if name == "plan":
        inputs = a.get("inputs")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            raise ValueError("plan: 'inputs' is required")
        options = [*opt("--profile", "profile"), *opt("--request", "request"), *opt("--name", "name"), *opt("--language", "language"), *_sets(a.get("set"))]
        options += [f"--kind={k}" for k in a.get("kinds") or []]
        if a.get("ai_brief", True):
            options += ["--ai-brief", *opt("--ai-model", "ai_model")]
        options += [*opt("--brief", "brief"), *_brief_file(a, tmpdir), *opt("--channel", "channel"), *(["--no-preferences"] if a.get("no_preferences") else []),
                    *opt("--entity-project", "entity_project"), *(["--no-entities"] if a.get("no_entities") else [])]
        if a.get("brief_requirements") is not None and not any(o.startswith("--ai-model=") for o in options):
            options += opt("--ai-model", "ai_model")
        return cmd("plan", options, [str(i) for i in inputs])
    if name == "brief":
        options = [*_sets(a.get("set")), *_brief_file(a, tmpdir)]
        options += opt("--ai-model", "ai_model") if a.get("brief_requirements") is not None else ["--ai-brief"]
        return cmd("brief", options, [need("text")])
    if name == "revise":
        options = [*opt("--feedback", "feedback"), *opt("--by", "by"), *_sets(a.get("set")), *opt("--scope", "scope"), *opt("--scope-by", "scope_by"), *opt("--channel", "channel")]
        if a.get("ai_recommendations") is not None:
            path = os.path.join(tmpdir, "ai_recommendations.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(a["ai_recommendations"], f, ensure_ascii=False)
            options += [f"--ai-recommendations={path}", *opt("--ai-model", "ai_model")]
        return cmd("revise", options, [need("project")])
    if name == "preferences":
        act = a.get("action") or "list"   # read-only by default
        if act == "promote":
            return ["preferences", "promote", f"--feedback={need('feedback')}", *opt("--confirm-scope", "confirm_scope"), *opt("--by", "by"), *opt("--reason", "reason"), "--", need("project")]
        if act in ("list", "show"):
            return ["preferences", act, *opt("--channel", "channel"), *(opt("--version", "version") if act == "show" else [])]
        if act == "diff":
            return ["preferences", "diff", *opt("--channel", "channel"), f"--from={need('from')}", f"--to={need('to')}"]
        if act == "rollback":
            return ["preferences", "rollback", *opt("--channel", "channel"), f"--to={need('to')}", *opt("--by", "by"), *opt("--reason", "reason")]
        raise ValueError("preferences: 'action' must be promote, list, show, diff or rollback")
    if name.startswith("entities_"):   # ADR-128
        act = name.split("_", 1)[1]
        proj = opt("--project", "project")

        def fields() -> List[str]:
            out = [*opt("--title", "title"), *opt("--note", "note"), *opt("--speaker-id", "speaker_id"), *opt("--voiceprint", "voiceprint")]
            for pr in a.get("pronunciation") or []:
                if not isinstance(pr, dict) or not pr.get("lang") or not pr.get("reading"):
                    raise ValueError(f"{name}: a pronunciation is {{lang, reading, ipa?, accent?}}")
                out.append(f"--pronunciation={pr['lang']}={pr['reading']}")
                out += [f"--ipa={pr['lang']}={pr['ipa']}"] if pr.get("ipa") else []
                out += [f"--accent={pr['lang']}={pr['accent']}"] if pr.get("accent") is not None else []
            return out
        if act == "list":
            return ["entities", "list", *proj, *opt("--type", "type")]
        if act == "add":
            return ["entities", "add", *proj, f"--type={need('type')}", *opt("--id", "id"), *fields(), *[f"--alias={x}" for x in a.get("aliases") or []], "--", need("canonical")]
        if act == "edit":
            return ["entities", "edit", *proj, *opt("--canonical", "canonical"), *opt("--type", "type"), *fields(), *[f"--add-alias={x}" for x in a.get("add_aliases") or []],
                    *[f"--remove-alias={x}" for x in a.get("remove_aliases") or []], *[f"--clear={x}" for x in a.get("clear") or []], "--", need("ref")]
        if act == "remove":
            return ["entities", "remove", *proj, "--", need("ref")]
        if act == "import":
            return ["entities", "import", *proj, *(["--replace"] if a.get("replace") else []), "--", need("file")]
        if act == "export":
            return ["entities", "export", *proj, *opt("--format", "format"), *opt("--output", "output")]
        if act == "suggest":
            return ["entities", "suggest", *proj, *opt("--min-count", "min_count"), "--", need("plan")]
        raise KeyError(name)
    if name.startswith("rename_"):   # ADR-129
        from .agent.rename import mcp_argv
        return mcp_argv(name, a)
    if name == "corrections":
        return cmd("corrections", [], [need("project")])
    if name in ("validate", "diff"):
        return cmd(name, [], [need("project")])
    if name == "explain":
        return cmd("explain", [*opt("--decision", "decision"), *(["--pipeline"] if a.get("pipeline") else [])], [need("project")])
    if name == "approve":
        return cmd("approve", [f"--decision={need('decision')}", *opt("--reason", "reason"), *opt("--by", "by")], [need("project")])
    if name == "reject":
        return cmd("reject", [f"--decision={need('decision')}", f"--reason={need('reason')}", *opt("--by", "by")], [need("project")])
    if name == "render":
        return cmd("render", [*(["--dry-run"] if a.get("dry_run") else []), *opt("--resume", "resume"), *(["--skip-preflight"] if a.get("skip_preflight") else [])],
                   [need("project")])
    if name == "preflight":
        return cmd("preflight", [], [need("project")])
    if name == "check":
        return cmd("check", [*opt("--platform", "platform"), *(["--qc"] if a.get("qc") else [])], [need("output")])
    if name == "artifacts":
        return cmd("artifacts", [], [str(a["project"])] if a.get("project") else [])
    if name == "selects":   # ADR-124: a preview; the render needs plan + approve
        ins = a.get("inputs")
        if not isinstance(ins, list) or not ins or not all(isinstance(x, str) and x for x in ins):
            raise ValueError("selects: 'inputs' must be a non-empty list of paths")
        options = [*opt("--max-duration", "max_duration"), *opt("--min-score", "min_score"), *opt("--max-blur", "max_blur"), *opt("--max-jitter", "max_jitter"),
                   *opt("--min-shot", "min_shot"), *(["--reject-dead-air"] if a.get("reject_dead_air") else []), *opt("--output", "output")]
        return cmd("selects", options, [str(x) for x in ins])
    if name == "transcript_edit":   # ADR-123: a preview; the cut needs plan + approve
        mode = a.get("mode")
        if mode not in ("remove", "keep"):
            raise ValueError("transcript_edit: 'mode' must be remove or keep")
        if not a.get("words") and not a.get("segments"):
            raise ValueError("transcript_edit: give 'words' and / or 'segments'")
        options = [f"--transcript={need('transcript')}", *opt(f"--{mode}-words", "words"), *opt(f"--{mode}-segments", "segments"),
                   *(["--include-overlap"] if a.get("include_overlap") else []), *[f"--subtitles={s}" for s in a.get("subtitles") or []], *opt("--output", "output")]
        return cmd("transcript-edit", options, [need("input")])
    raise KeyError(name)


class Server:
    def __init__(self, workspace: Optional[str] = None, ffmpeg_skill_dir: Optional[str] = None, python: str = sys.executable,
                 permissions: Optional[Permissions] = None, clock: Callable[[], float] = time.monotonic):
        self.global_args = ["--json", *(["--workspace", workspace] if workspace else []), *(["--ffmpeg-skill-dir", ffmpeg_skill_dir] if ffmpeg_skill_dir else [])]
        self.python = python
        # the child resolves the workspace the same way (service.Service), from the same working directory
        self.workspace = str(Path(workspace or os.environ.get("VIDEO_AGENT_WORKSPACE") or "./video-agent-work").resolve())
        self.permissions = permissions or load_permissions(self.workspace)
        self.tokens = ConfirmationTokens(self.permissions.confirm_ttl_s, clock)

    def _gate(self, name: str, arguments: Dict[str, Any], token: Optional[str]):
        """(call, None) when the call may run now; (None, result) for a refusal or a preview."""
        perms = self.permissions
        try:
            call = classify(name, arguments, perms, self.workspace)
        except KeyError:
            if any(t["name"] == name for t in TOOLS):
                return None, _refusal("NO_PERMISSION_CLASS", name, None, f"{name} has no permission class on this server, so it is refused")
            return None, _text(f"unknown tool: {name}", error=True)
        if not perms.enabled(call.cls):
            return None, _refusal("SCOPE_DISABLED", name, call.cls, f"this {name} call is {call.cls}, which is disabled on this server",
                                  enable_with=enable_hint(call.cls), enabled_scopes=list(perms.scopes), would_do=call.preview or None)
        outside = [p for p in call.writes if not perms.inside(p)]
        if outside:
            return None, _refusal("PATH_OUTSIDE_ROOTS", name, call.cls, "the call writes outside the allowed roots (symlinks and '..' resolved)",
                                  paths=[{"path": p, "resolved": os.path.realpath(p)} for p in outside], allowed_roots=list(perms.roots),
                                  enable_with="start the server with --allow-root DIR (or VIDEO_AGENT_MCP_ROOTS, or \"allowed_roots\" in the config file)")
        if call.cls not in CONFIRMED_CLASSES:
            return call, None
        digest = call_digest(name, arguments, call)
        if token is None:
            tok = self.tokens.mint(digest)
            body = {"status": "CONFIRMATION_REQUIRED", "tool": name, "class": call.cls, "preview": call.preview,
                    "writes": [os.path.realpath(p) for p in call.writes], "confirm_token": tok, "expires_in_s": self.permissions.confirm_ttl_s,
                    "next": f"show the user this preview; only after they agree, call {name} again with the same arguments and confirm_token "
                            "(single use; any change to the arguments or to the files listed needs a new preview)"}
            return None, _text(json.dumps(body, ensure_ascii=False, indent=2), error=False)
        code = self.tokens.redeem(token, digest)
        if code:
            why = {"CONFIRM_TOKEN_UNKNOWN": "the token was never issued here, or was already used (tokens are single use)",
                   "CONFIRM_TOKEN_EXPIRED": f"the token expired ({self.permissions.confirm_ttl_s:g} s after its preview)",
                   "CONFIRM_TOKEN_MISMATCH": "the token was issued for a different call, or the files it previewed changed since"}[code]
            return None, _refusal(code, name, call.cls, why + "; nothing was done. Call without confirm_token for a new preview")
        return call, None

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        arguments = dict(arguments or {})
        token = arguments.pop("confirm_token", None)
        call, refused = self._gate(name, arguments, token)
        if refused is not None:
            return refused
        env = None
        if self.permissions.paid_ai_provider and not call.spends:   # a call not confirmed as spending never reaches a paid provider
            env = dict(os.environ, **{ENV_AI_PROVIDER: "null"})
        with tempfile.TemporaryDirectory(prefix="video-agent-mcp-") as tmp:
            try:
                argv = tool_argv(name, arguments, tmp)
            except KeyError:
                return _text(f"unknown tool: {name}", error=True)
            except ValueError as exc:
                return _text(str(exc), error=True)
            try:
                p = subprocess.run(safe_command([*python_module_command("video_agent.cli", self.python), *self.global_args, *argv]), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=TOOL_TIMEOUT_S, stdin=subprocess.DEVNULL, **({"env": env} if env else {}))
            except subprocess.TimeoutExpired:
                return _text(f"{name}: timed out after {TOOL_TIMEOUT_S} s", error=True)
        out = p.stdout.strip()
        if p.stderr.strip():
            out += ("\n\n" if out else "") + "stderr:\n" + p.stderr.strip()[-4000:]
        # 2 validation errors / 3 BLOCKED / 4 waiting for approval / 5 QA FAIL or no new version are answers to show the user, not tool failures
        return _text(out if p.returncode == 0 else f"exit code {p.returncode}\n{out}", error=p.returncode not in (0, 2, 3, 4, 5))

    def handle(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:   # a notification (initialized, cancelled, ...): no reply
            return None
        if method == "initialize":
            result: Any = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                           "serverInfo": {"name": "video-agent", "version": __version__}}
        elif method == "tools/list":
            result = {"tools": listed_tools(self.permissions)}
        elif method == "tools/call":
            params = msg.get("params") or {}
            result = self.call_tool(str(params.get("name")), params.get("arguments") or {})
        elif method == "ping":
            result = {}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, stdin=None, stdout=None) -> None:
        stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                reply: Optional[Dict[str, Any]] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                try:
                    reply = self.handle(msg) if isinstance(msg, dict) else {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid request"}}
                except Exception as exc:  # noqa: BLE001 -- a tool bug must not end the session
                    reply = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32000, "message": str(exc)}}
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def _text(text: str, error: bool = False) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def _refusal(code: str, tool: str, cls: Optional[str], message: str, **extra: Any) -> Dict[str, Any]:
    body = {"error": code, "tool": tool, "class": cls, "message": message, **{k: v for k, v in extra.items() if v is not None}}
    return _text(json.dumps(body, ensure_ascii=False, indent=2), error=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="video-agent-mcp", description="video-agent as an MCP server (stdio)")
    ap.add_argument("--workspace", help="workspace root (default $VIDEO_AGENT_WORKSPACE or ./video-agent-work)")
    ap.add_argument("--ffmpeg-skill-dir")
    ap.add_argument("--scopes", help="enabled permission classes, comma separated: read, write, destructive, publish, or all "
                                     "(default $VIDEO_AGENT_MCP_SCOPES, the config file, else read,write)")
    ap.add_argument("--allow-root", action="append", metavar="DIR", help="an absolute directory writing calls may write in, besides the workspace "
                                                                          "(repeatable; default $VIDEO_AGENT_MCP_ROOTS or the config file)")
    ap.add_argument("--config", help="JSON file with scopes, allowed_roots and confirm_ttl_s (default $VIDEO_AGENT_MCP_CONFIG)")
    ap.add_argument("--confirm-ttl", type=float, metavar="S", help="lifetime of a confirm_token in seconds (default 300)")
    a = ap.parse_args(argv)
    workspace = str(Path(a.workspace or os.environ.get("VIDEO_AGENT_WORKSPACE") or "./video-agent-work").resolve())
    try:
        perms = load_permissions(workspace, a.scopes, a.allow_root, a.config, a.confirm_ttl)
    except PermissionConfigError as exc:
        ap.error(str(exc))
    Server(a.workspace, a.ffmpeg_skill_dir, permissions=perms).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
