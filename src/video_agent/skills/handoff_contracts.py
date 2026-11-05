"""Pinned contracts of the Skills a handed-off job names (ADR-080, on ADR-057): what the agent tells the host to run, and nothing it runs.

ADR-057 declared one phase-2 SkillSpec per ecosystem job with a tool id named after the job (`blender/render`), because no adapter had
read the owning Skill's contract yet. For a job listed here the SkillSpec cites the Skill's own `skill_id` and the `provides[].tool_id`
of a contract pinned in `skills/pins/<skill_id>_<version>.json` (the Skill's `contract --json` from its main, keys sorted, one line), and
the job takes typed parameters, `job.<skill>.<name>`, whose vocabulary (enum values) is read from that pin. The plan step still has no
tool and the plan is still BLOCKED: there is no adapter, so the agent never starts the Skill, never writes its output and never admits
the result into the Project IR. What changes is the handoff: it names the exact tool, the arguments by the contract's own names, what
is still missing, and where the output goes next in this agent's pipeline.

Each parameter is checked before analysis (a value outside the pinned vocabulary, a file that is not there, a parameter without the
`job.<skill>=true` switch is refused, never guessed). File / folder parameters are resolved to absolute paths; nothing is read from them
beyond their existence (and, for a scene folder, its index.html)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PINS_DIR = Path(__file__).with_name("pins")
_ON = (True, "true", "yes", "on", "1")
_OFF = (False, "false", "no", "off", "0")


class HandoffParameterError(ValueError):
    pass


def load_pin(name: str) -> Dict[str, Any]:
    return json.loads((PINS_DIR / name).read_text(encoding="utf-8"))


def _tool_entry(contract: Dict[str, Any], tool: str) -> Dict[str, Any]:
    """The contract's entry for a tool id (blender-skill names it `tool_id`, hyperframes-skill `id`)."""
    for t in contract.get("tools") or []:
        if isinstance(t, dict) and tool in (t.get("tool_id"), t.get("id")):
            return t
    raise KeyError(f"pinned contract has no tool {tool}")


@dataclass(frozen=True)
class HandoffParam:
    name: str                                   # requirement key suffix: job.<skill>.<name>
    arg: str                                    # the contract's name for it (input_schema property, or the flag / positional it documents)
    kind: str                                   # file | dir | enum | bool
    choices: Callable[[Dict[str, Any]], Tuple[str, ...]] = lambda c: ()   # enum values, read from the pinned contract
    suffixes: Tuple[str, ...] = ()              # file: accepted extensions (lower case)
    marker: Optional[str] = None                # dir: a file that must be inside it
    required: bool = False                      # the tool cannot run without it (the handoff lists it as missing when not given)
    doc: str = ""


@dataclass(frozen=True)
class HandoffPin:
    skill: str                                  # the agent's SkillSpec name
    package: str                                # the Skill's own skill_id (prefix of its tool ids)
    version: str                                # the pinned contract's version
    pin_file: str                               # skills/pins/<file>
    source: str                                 # where the pin was read from
    tool: str                                   # provides[].tool_id the job is
    params: Tuple[HandoffParam, ...]
    next_step: Callable[[Dict[str, Any]], str]  # where the output goes next in this agent, given the parameters asked for
    note: str = ""

    def contract(self) -> Dict[str, Any]:
        return load_pin(self.pin_file)

    def param(self, name: str) -> Optional[HandoffParam]:
        return next((p for p in self.params if p.name == name), None)

    def provides(self) -> Dict[str, Any]:
        c = self.contract()
        return next(p for p in c.get("provides") or [] if p.get("tool_id") == self.tool)


def _blender_presets(c: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(_tool_entry(c, "blender-skill/render")["options"]["--resolution-preset"]["choices"])


def _hyperframes_codecs(c: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(_tool_entry(c, "hyperframes-skill/render")["input_schema"]["properties"]["codec"]["enum"])


def _render_3d_next(args: Dict[str, Any]) -> str:
    if args.get("transparent"):
        return ("a B-roll layer with alpha (ProRes 4444 .mov, or an RGBA PNG sequence whose `sequence` block is ffmpeg-skill/sequence's interface) "
                "to composite over the programme with ffmpeg-skill/overlay; .mp4 / .jpg are refused with transparent")
    return "a B-roll clip (turntable .mp4) or still (.png) to place over the programme with ffmpeg-skill/overlay"


def _html_scene_next(args: Dict[str, Any]) -> str:
    if args.get("codec") == "prores":
        return "a ProRes 4444 .mov with alpha to composite over the programme with ffmpeg-skill/overlay"
    return "an opaque clip (.mp4 h264 / .webm vp9) to join before or after the programme with video-editing/concat (edit.concat)"


# blender-skill's own list of the 3D files it reads (SKILL.md description: glb/gltf, fbx, obj, stl, usd/usdz, ply, abc, .blend)
_BLENDER_MODELS = (".glb", ".gltf", ".fbx", ".obj", ".stl", ".usd", ".usda", ".usdc", ".usdz", ".ply", ".abc", ".blend")

PINS: Dict[str, HandoffPin] = {p.skill: p for p in (
    HandoffPin(
        skill="render_3d", package="blender-skill", version="0.31.0", pin_file="blender-skill_0.31.0.json",
        source="kajisho5/blender-skill main befc3e3 (`python3 scripts/_contract.py --json`; = the blender-skill-v0.31.0 contract)",
        tool="blender-skill/render",
        params=(
            HandoffParam("model", "file", "file", suffixes=_BLENDER_MODELS, required=True, doc="the 3D file to render (positional FILE)"),
            HandoffParam("preset", "--resolution-preset", "enum", choices=_blender_presets, doc="frame size by name (the contract's resolution presets)"),
            HandoffParam("transparent", "--transparent", "bool", doc="render with alpha for compositing (SKILL.md: RGBA PNG or ProRes 4444 .mov)"),
        ),
        next_step=_render_3d_next,
        note="needs Blender 4.2+ on the machine (the contract's external_tools.required); `python3 scripts/_contract.py doctor --json` says whether it is found"),
    HandoffPin(
        skill="html_scene_render", package="hyperframes-skill", version="0.6.2", pin_file="hyperframes-skill_0.6.2.json",
        source="kajisho5/hyperframes-skill main 9d16f7f (= v0.6.2; `node bin/hyperframes-skill.mjs contract --json`)",
        tool="hyperframes-skill/render",
        params=(
            HandoffParam("scene", "scene_dir", "dir", marker="index.html", required=True, doc="the scene directory with index.html"),
            HandoffParam("codec", "codec", "enum", choices=_hyperframes_codecs, doc="h264 (.mp4), vp9 (.webm) or prores (.mov, ProRes 4444 with alpha)"),
        ),
        next_step=_html_scene_next,
        note="needs Node.js 22+, HyperFrames and a headless Chromium (the contract's render capabilities); the render records the argv of every "
             "process it starts (execution.argv_recorder) and cross-checks its output with ffmpeg-skill/probe when a checkout is present"),
)}


def pin(skill: str) -> Optional[HandoffPin]:
    return PINS.get(skill)


def parse_param(p: HandoffPin, name: str, value: Any) -> Any:
    """The checked value of `job.<skill>.<name>` (a path made absolute, a bool, an enum value), or HandoffParameterError naming the
    accepted values. Enum values come from the pinned contract, never from this module."""
    key = f"job.{p.skill}.{name}"
    hp = p.param(name)
    if hp is None:
        raise HandoffParameterError(f"unknown parameter {key!r}; {p.skill} takes " + ", ".join(f"job.{p.skill}.{x.name}" for x in p.params))
    if hp.kind == "bool":
        v = value.strip().lower() if isinstance(value, str) else value
        if v in _ON:
            return True
        if v in _OFF:
            return False
        raise HandoffParameterError(f"{key} is a switch (true / false), got {value!r}")
    if not isinstance(value, str) or not value.strip():
        raise HandoffParameterError(f"{key} must be a non-empty string, got {value!r}")
    if hp.kind == "enum":
        choices = hp.choices(p.contract())
        if value not in choices:
            raise HandoffParameterError(f"{key} must be one of {', '.join(choices)} ({p.package} {p.version} contract), got {value!r}")
        return value
    path = Path(value).expanduser()
    if hp.kind == "file":
        if not path.is_file():
            raise HandoffParameterError(f"{key}: no such file {value!r}")
        if hp.suffixes and path.suffix.lower() not in hp.suffixes:
            raise HandoffParameterError(f"{key}: {path.name} is not a file {p.package} reads ({', '.join(hp.suffixes)})")
    else:
        if not path.is_dir():
            raise HandoffParameterError(f"{key}: no such folder {value!r}")
        if hp.marker and not (path / hp.marker).is_file():
            raise HandoffParameterError(f"{key}: {value!r} has no {hp.marker}")
    return str(path.resolve())


def handoff_request(skill: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The handoff's `contract` block: the pinned Skill, the tool, the arguments by the contract's names, what the tool still needs, and
    where its output goes next here. None for a job without a pin. Data for the host; the agent runs nothing."""
    p = PINS.get(skill)
    if p is None:
        return None
    prov = p.provides()
    args = {hp.arg: params[hp.name] for hp in p.params if hp.name in params}
    missing = [f"job.{skill}.{hp.name} ({hp.doc})" for hp in p.params if hp.required and hp.name not in params]
    return {"package": p.package, "version": p.version, "source": p.source, "tool": p.tool, "capability": prov.get("id"), "lifecycle": prov.get("lifecycle"),
            "args": args, "missing": missing, "output": "chosen by whoever runs the tool, never the input; the agent does not admit it into the plan",
            "next": p.next_step(params), "note": p.note}


def check_pin_against(contract: Dict[str, Any], p: HandoffPin) -> List[str]:
    """What the handoff relies on in a live contract (a sibling checkout's): the identity, the tool in provides[], and every enum read
    from the pin. Non-empty: the pin is stale and must be taken again."""
    errs: List[str] = []
    sid = contract.get("skill_id") or (contract.get("skill") or {}).get("id")
    if sid != p.package:
        errs.append(f"skill_id {sid!r} != {p.package}")
    if not any(isinstance(x, dict) and x.get("tool_id") == p.tool for x in contract.get("provides") or []):
        errs.append(f"provides[] has no {p.tool}")
    pinned = p.contract()
    for hp in p.params:
        if hp.kind == "enum":
            try:
                live = hp.choices(contract)
            except (KeyError, TypeError) as e:
                errs.append(f"{hp.name}: not in the live contract ({e})")
                continue
            if live != hp.choices(pinned):
                errs.append(f"{hp.name}: pinned {hp.choices(pinned)} != live {live}")
    return errs
