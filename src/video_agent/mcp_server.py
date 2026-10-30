"""MCP server (ADR-050): the video-agent CLI as tools for a chat app -- Claude Desktop, Claude Code, any MCP client.

Stdio JSON-RPC 2.0, standard library only. Each tool runs `video-agent --json <command> ...` in a child process, so
the protocol stream on stdout never sees a skill's output, and the result is exactly what the CLI prints. The chat
model is the reasoning step (providers/host.py): `plan` returns an ai_brief, the model answers it, and `revise` takes
that answer as `ai_recommendations`, validated like any provider's. Approval and rendering stay separate tools: the
server never approves on its own, and `render` has no inline approve.

    python -m video_agent.mcp_server [--workspace DIR] [--ffmpeg-skill-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from . import __version__

PROTOCOL_VERSION = "2024-11-05"
TOOL_TIMEOUT_S = 6 * 3600   # a render of a long recording can take hours; the client may cancel sooner

_STR = {"type": "string"}
_PROJECT = {"type": "string", "description": "path of the project.json that plan printed"}

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
         "kinds": {"type": "array", "items": _STR, "description": "extra analysis kinds, e.g. transcript, scene_detection"},
         "ai_brief": {"type": "boolean", "description": "return the brief for the chat model (default true)"},
         "ai_model": {"type": "string", "description": "name of the chat model, recorded in provenance"}}}},
    {"name": "revise", "description": (
        "Write the next plan version. Pass the chat model's answer to the ai_brief as ai_recommendations "
        "({\"recommendations\": [{intent, asset_id, statement, confidence, evidence: [ids], params}]}); every one is "
        "validated (cited ids must exist, unknown intents are dropped) and the result lists accepted and dropped ones. "
        "Also takes the user's feedback after a reject."),
     "inputSchema": {"type": "object", "required": ["project"], "properties": {
         "project": _PROJECT, "feedback": {"type": "string"}, "set": {"type": "object"}, "by": _STR,
         "ai_recommendations": {"type": ["object", "array"], "description": "the answer to the ai_brief"},
         "ai_model": {"type": "string", "description": "name of the chat model that answered"}}}},
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
    {"name": "render", "description": "Execute an approved plan through the skills, then QA. dry_run shows the commands without running them. Pending CONFIRM decisions block it: approve them first.",
     "inputSchema": {"type": "object", "required": ["project"], "properties": {
         "project": _PROJECT, "dry_run": {"type": "boolean"}, "resume": {"type": "string", "description": "job id or last"}}}},
    {"name": "check", "description": "Probe an output file and check platform compliance (loudness, codec, resolution).",
     "inputSchema": {"type": "object", "required": ["output"], "properties": {
         "output": _STR, "platform": {"type": "string", "description": "e.g. youtube (default custom)"}, "qc": {"type": "boolean"}}}},
    {"name": "artifacts", "description": "Registered artifacts of a project (or all), with QA status.",
     "inputSchema": {"type": "object", "properties": {"project": _PROJECT}}},
]


def _sets(values: Optional[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for k, v in (values or {}).items():
        out += ["--set", f"{k}={json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}"]
    return out


def tool_argv(name: str, a: Dict[str, Any], tmpdir: str) -> List[str]:
    """CLI arguments for a tool call (after the global options). Raises ValueError on a missing argument."""
    def need(key: str) -> str:
        v = a.get(key)
        if v in (None, "", []):
            raise ValueError(f"{name}: '{key}' is required")
        return str(v)

    def opt(flag: str, key: str) -> List[str]:
        return [flag, str(a[key])] if a.get(key) not in (None, "") else []

    if name in ("doctor", "skills"):
        return [name]
    if name == "plan":
        inputs = a.get("inputs")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not inputs:
            raise ValueError("plan: 'inputs' is required")
        argv = ["plan", *map(str, inputs), *opt("--profile", "profile"), *opt("--request", "request"), *opt("--name", "name"),
                *opt("--language", "language"), *_sets(a.get("set"))]
        for k in a.get("kinds") or []:
            argv += ["--kind", str(k)]
        if a.get("ai_brief", True):
            argv += ["--ai-brief", *opt("--ai-model", "ai_model")]
        return argv
    if name == "revise":
        argv = ["revise", need("project"), *opt("--feedback", "feedback"), *opt("--by", "by"), *_sets(a.get("set"))]
        if a.get("ai_recommendations") is not None:
            path = os.path.join(tmpdir, "ai_recommendations.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(a["ai_recommendations"], f, ensure_ascii=False)
            argv += ["--ai-recommendations", path, *opt("--ai-model", "ai_model")]
        return argv
    if name in ("validate", "diff"):
        return [name, need("project")]
    if name == "explain":
        return ["explain", need("project"), *opt("--decision", "decision"), *(["--pipeline"] if a.get("pipeline") else [])]
    if name == "approve":
        return ["approve", need("project"), "--decision", need("decision"), *opt("--reason", "reason"), *opt("--by", "by")]
    if name == "reject":
        return ["reject", need("project"), "--decision", need("decision"), "--reason", need("reason"), *opt("--by", "by")]
    if name == "render":
        return ["render", need("project"), *(["--dry-run"] if a.get("dry_run") else []), *opt("--resume", "resume")]
    if name == "check":
        return ["check", need("output"), *opt("--platform", "platform"), *(["--qc"] if a.get("qc") else [])]
    if name == "artifacts":
        return ["artifacts", *([str(a["project"])] if a.get("project") else [])]
    raise KeyError(name)


class Server:
    def __init__(self, workspace: Optional[str] = None, ffmpeg_skill_dir: Optional[str] = None, python: str = sys.executable):
        self.global_args = ["--json", *(["--workspace", workspace] if workspace else []), *(["--ffmpeg-skill-dir", ffmpeg_skill_dir] if ffmpeg_skill_dir else [])]
        self.python = python

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="video-agent-mcp-") as tmp:
            try:
                argv = tool_argv(name, arguments or {}, tmp)
            except KeyError:
                return _text(f"unknown tool: {name}", error=True)
            except ValueError as exc:
                return _text(str(exc), error=True)
            try:
                p = subprocess.run([self.python, "-m", "video_agent.cli", *self.global_args, *argv], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=TOOL_TIMEOUT_S, stdin=subprocess.DEVNULL)
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
            result = {"tools": TOOLS}
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="video-agent-mcp", description="video-agent as an MCP server (stdio)")
    ap.add_argument("--workspace", help="workspace root (default $VIDEO_AGENT_WORKSPACE or ./video-agent-work)")
    ap.add_argument("--ffmpeg-skill-dir")
    a = ap.parse_args(argv)
    Server(a.workspace, a.ffmpeg_skill_dir).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
