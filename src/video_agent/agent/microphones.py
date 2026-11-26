"""Microphone tracks of a clip (ADR-134): what a source's microphones are, and how the agent mixes and transcribes them.

A conference clip often carries one microphone per person: separate files from the recorder, or one channel each of the
camera's / the desk's own multichannel audio. The Project IR records them on the clip's asset (`assets[<id>].microphones`)
and the plan lowers them through the Skills that already do the work:

    microphone files / channels ─(the engine's audio tool: one channel, the clip's rate)─→ one mono WAV per microphone
        ─(audio-production AUTOMIX)─→ the floor mix ─(the engine's audio tool: replace the sound)─→ the clip with the floor mix as its sound
        [─(audio-production MIX_TRACKS, languages)─→ one language-tagged track per interpreter language, beside the programme]

and, when the plan measures a transcript of that clip, `transcription/microphones` transcribes each microphone on its own and
merges them by time, the microphone's name as the speaker (no diarization); words said over someone else are kept under
`overlapping_segments` and recorded by the plan.

Requirements (explicit only; nothing here is switched on by a profile):

- `audio.microphones` (JSON list, or a JSON file of one): [{id?, name, speaker?, language?, role?, path | channel, audio_stream?, pinned?,
  weight_db?, gain_db?, pan?, clip?}]. `path` is a microphone file (a separate recording of the same take, starting with the clip),
  `channel` a 0-based channel of the clip's own audio. role: floor (default: a panel microphone, shared in the automix), presenter
  (a lectern / presenter microphone, in the automix like floor; `pinned` keeps it open) or interp (an interpreter: not in the
  automix and not transcribed, the input of its language's track). `language` is the ISO 639-2 code (eng, jpn) MIX_TRACKS writes as the
  track's tag; an interp microphone needs it. `clip` names the input the microphones belong to (path or file name) when there are several.
- `audio.automix` = true | <preset name>: the gain-sharing automatic mix of the floor / presenter microphones.
  `audio.automix.window_ms`, `audio.automix.max_gain_rate_db_per_s`, `audio.automix.hold_below_dbfs`: the Skill's own parameters.
- `audio.language_tracks` = true: one track per interpreter language (the floor under each interpreter), written as an MKV of
  language-tagged tracks next to the programme; `audio.language_tracks.floor_gain_db`: the floor's level under the interpreter.

Production values (ADR-134): AUTOMIX has no default for `window_ms` and `max_gain_rate_db_per_s` (audio-production's rule: no
silent production default). The agent keeps its values as named presets in the profile (`automix_presets`, generic.json), each
value explicit, and records the preset's values as PROFILE requirements of the plan (`requirements.preset_requirements`), so
every value in the plan says where it came from; an explicit `--set` of a value wins. There is no value in this module.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import Requirement
from .editing import EditRequirementError, _bool, _number

# ---- vocabulary ------------------------------------------------------------------------------------------------------
MICS_KEY = "audio.microphones"
AUTOMIX_KEY = "audio.automix"
WINDOW_KEY = "audio.automix.window_ms"
RATE_KEY = "audio.automix.max_gain_rate_db_per_s"
HOLD_KEY = "audio.automix.hold_below_dbfs"
PRESET_KEY = "audio.automix.preset"          # the preset the plan used (PROFILE requirement written by requirements.preset_requirements)
TRACKS_KEY = "audio.language_tracks"
FLOOR_KEY = "audio.language_tracks.floor_gain_db"
FLOOR_LANGUAGE_KEY = "audio.language_tracks.floor_language"
USER_KEYS = (MICS_KEY, AUTOMIX_KEY, WINDOW_KEY, RATE_KEY, HOLD_KEY, TRACKS_KEY, FLOOR_KEY, FLOOR_LANGUAGE_KEY)
REQUIREMENT_KEYS = USER_KEYS + (PRESET_KEY,)
PATH_KEYS = (MICS_KEY,)                       # names files: never a brief's or a stored preference's key
# preset field → requirement key (profile `automix_presets.<name>`)
PRESET_FIELDS = {"window_ms": WINDOW_KEY, "max_gain_rate_db_per_s": RATE_KEY, "hold_below_dbfs": HOLD_KEY, "floor_gain_db": FLOOR_KEY}
PRESETS_DATA_KEY = "automix_presets"
DEFAULT_PRESET_FIELD = "default"

SUBJECT = "audio.automix"
TRACKS_SUBJECT = "audio.language_tracks"
TRANSCRIPT_SUBJECT = "speech.microphones"
APPROVAL_KEY = ("audio.automix.approval", "CONFIRM")
TRACKS_APPROVAL_KEY = ("audio.language_tracks.approval", "CONFIRM")

EXTRACT_SKILL = "mic_extract"                 # ffmpeg-skill/audio: one microphone as a mono WAV at the clip's sample rate
AUTOMIX_SKILL = "audio_automix"               # audio-production AUTOMIX
TRACKS_SKILL = "audio_mix_tracks"             # audio-production MIX_TRACKS (languages)
TRACKS_FORMAT = "mkv"                         # PCM tracks, each with its language tag and name (MIX_TRACKS track set)
TRACKS_ROLE = "LANGUAGE_TRACKS"

ROLES = ("floor", "presenter", "interp")
MIXED_ROLES = ("floor", "presenter")
MIC_FIELDS = ("id", "name", "speaker", "language", "role", "path", "channel", "audio_stream", "pinned", "weight_db", "gain_db", "pan", "clip")
# the pinned audio-production 1.5.0 contract's ranges (tests/test_microphones.py checks them against the pin)
AUTOMIX_INPUTS = (2, 8)
TRACKS_INPUTS = (2, 8)
WINDOW_RANGE = (1.0, 1000.0)
RATE_RANGE = (10.0, 10000.0)
HOLD_RANGE = (-100.0, 0.0)
GAIN_RANGE = (-60.0, 60.0)
WEIGHT_RANGE = (-30.0, 30.0)
PAN_RANGE = (-1.0, 1.0)
MAX_MICS = 16
MAX_CHANNEL = 63
# the request's / brief's words for the two switches (English and Japanese, after NFKC); read only for a clip whose microphones are declared
AUTOMIX_PHRASES = re.compile(r"\bauto[- ]?mix(?:ing|ed)?\b|\bautomatic\s+mix(?:ing)?\b|\bgain[- ]sharing\b|\bpanel\s+(?:audio|sound|mics?|microphones)\b"
                             r"|オートミックス|オートミキサー|自動ミックス|自動ミキシング|パネル(?:の)?(?:音|音声|マイク)", re.I)
TRACKS_PHRASES = re.compile(r"\binterpret(?:er|ation|ing)\s+(?:tracks?|audio|channels?|feeds?)\b|\blanguage\s+tracks?\b|\bper[- ]language\s+(?:tracks?|audio)\b"
                            r"|通訳(?:の)?(?:トラック|音声|チャンネル)|同時通訳|言語別(?:の)?(?:トラック|音声)|多言語(?:の)?(?:トラック|音声)", re.I)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_LANG_RE = re.compile(r"^[a-z]{3}$")
# transcription-skill's microphone names (its microphones.NAME_RE): what is sent as the speaker id; any other spelling is kept as the display name
TRANSCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


def _text(v: Any, what: str, limit: int = 64) -> str:
    if not isinstance(v, str) or not v.strip() or len(v) > limit or any(unicodedata.category(c).startswith("C") for c in v):
        raise EditRequirementError(f"{what} must be printable text of 1..{limit} characters, got {v!r}")
    return v.strip()


def _load_list(value: Any) -> List[Any]:
    """`audio.microphones` as given: a JSON list, its JSON text, or the path of a JSON file holding one."""
    v = value
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                v = json.loads(s)
            except ValueError as e:
                raise EditRequirementError(f"{MICS_KEY} is not valid JSON: {e}")
        elif s.lower().endswith(".json") and os.path.isfile(s):
            try:
                v = json.loads(Path(s).read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise EditRequirementError(f"{MICS_KEY}: {s} is not a readable JSON file: {e}")
    if not isinstance(v, list) or not v:
        raise EditRequirementError(f"{MICS_KEY} must be a non-empty JSON list of microphones [{{name, path | channel, role?, ...}}] (or a .json file of one), got {value!r}")
    return v


def parse_microphones(value: Any) -> List[Dict[str, Any]]:
    """The declared microphones, normalised (ids, roles, absolute paths). Each one is either a file or a channel of the clip, never both."""
    raw = _load_list(value)
    if len(raw) > MAX_MICS:
        raise EditRequirementError(f"{MICS_KEY}: at most {MAX_MICS} microphones, got {len(raw)}")
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(raw):
        what = f"{MICS_KEY}[{i}]"
        if not isinstance(m, dict):
            raise EditRequirementError(f"{what} must be an object {{name, path | channel, ...}}")
        unknown = sorted(k for k in m if k not in MIC_FIELDS)
        if unknown:
            raise EditRequirementError(f"{what}: unknown field(s) {unknown} (fields: {', '.join(MIC_FIELDS)})")
        mid = str(m.get("id") if m.get("id") is not None else f"mic{i + 1}")
        if not _ID_RE.match(mid):
            raise EditRequirementError(f"{what}.id must be 1..32 letters, digits, '_' or '-', got {mid!r}")
        rec: Dict[str, Any] = {"id": mid, "name": _text(m.get("name") if m.get("name") is not None else mid, f"{what}.name")}
        if m.get("speaker") is not None:
            rec["speaker"] = _text(m["speaker"], f"{what}.speaker", 100)
        role = str(m.get("role") or "floor").strip().lower()
        if role not in ROLES:
            raise EditRequirementError(f"{what}.role must be one of {ROLES}, got {m.get('role')!r}")
        rec["role"] = role
        if m.get("language") is not None:
            lang = str(m["language"]).strip().lower()
            if not _LANG_RE.match(lang):
                raise EditRequirementError(f"{what}.language must be an ISO 639-2 code (three lowercase letters: eng, jpn, ...), got {m['language']!r}")
            rec["language"] = lang
        if role == "interp" and "language" not in rec:
            raise EditRequirementError(f"{what}: an interpreter microphone needs its language (ISO 639-2), the tag of its track")
        has_path, has_channel = m.get("path") is not None, m.get("channel") is not None
        if has_path == has_channel:
            raise EditRequirementError(f"{what} needs exactly one of path (a microphone file) or channel (a channel of the clip's own audio)")
        if has_path:
            p = m["path"]
            if not isinstance(p, str) or not p.strip() or "\x00" in p or "\n" in p:
                raise EditRequirementError(f"{what}.path must be a file path")
            ap = os.path.abspath(p)
            if not os.path.isfile(ap):
                raise EditRequirementError(f"{what}.path {p!r} is not a file")
            rec["path"] = ap
            if m.get("audio_stream") is not None:
                rec["audio_stream"] = int(_int(m["audio_stream"], f"{what}.audio_stream", 0, 31))
        else:
            if m.get("audio_stream") is not None:
                raise EditRequirementError(f"{what}.audio_stream is for a microphone file; a channel is of the clip's first audio stream")
            rec["channel"] = int(_int(m["channel"], f"{what}.channel", 0, MAX_CHANNEL))
        if m.get("pinned") is not None:
            rec["pinned"] = _bool(m["pinned"], f"{what}.pinned")
        for k, rng in (("weight_db", WEIGHT_RANGE), ("gain_db", GAIN_RANGE), ("pan", PAN_RANGE)):
            if m.get(k) is not None:
                rec[k] = float(_number(m[k], f"{what}.{k}", rng[0], rng[1]))
        if role == "interp" and (rec.get("pinned") is not None or "weight_db" in rec):
            raise EditRequirementError(f"{what}: pinned / weight_db are automix settings; an interpreter microphone is not in the automix")
        if role != "interp" and "pan" in rec:
            raise EditRequirementError(f"{what}: AUTOMIX writes one mono floor mix, so a floor / presenter microphone has no pan (pan an interpreter in its track)")
        if m.get("clip") is not None:
            rec["clip"] = _text(m["clip"], f"{what}.clip", 1024)
        out.append(rec)
    ids = [r["id"] for r in out]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        raise EditRequirementError(f"{MICS_KEY}: microphone id(s) {dup} given twice")
    srcs = [(r.get("clip"), r.get("path"), r.get("audio_stream", 0)) if "path" in r else (r.get("clip"), "channel", r["channel"]) for r in out]
    if len(set(srcs)) != len(srcs):
        raise EditRequirementError(f"{MICS_KEY}: the same file / channel is given as two microphones")
    return out


def _int(v: Any, what: str, lo: int, hi: int) -> int:
    x = _number(v, what, lo, hi)
    if float(x) != int(x):
        raise EditRequirementError(f"{what} must be an integer, got {v!r}")
    return int(x)


def _given(m: Dict[str, Requirement], key: str) -> Optional[Requirement]:
    r = m.get(key)
    return r if r is not None and r.provenance != "DEFAULT" else None


def parse_requirements(m: Dict[str, Requirement], user_only: bool = False) -> Optional[Dict[str, Any]]:
    """Requirement map → {microphones, automix: bool, preset, window_ms?, max_gain_rate_db_per_s?, hold_below_dbfs?, tracks: bool,
    floor_gain_db?, requirements} or None when no microphone key is set. A refinement without its switch is an ambiguous request and is refused."""
    given = {k: _given(m, k) for k in REQUIREMENT_KEYS}
    asked = [k for k in USER_KEYS if given[k] is not None and (not user_only or given[k].provenance == "USER")]
    if not asked:
        return None
    if given[MICS_KEY] is None:
        raise EditRequirementError(f"{asked[0]} is set but no microphones are declared: give {MICS_KEY}=[{{name, path | channel, role?}}, ...] "
                                   "(one entry per microphone of the clip)")
    out: Dict[str, Any] = {"microphones": parse_microphones(given[MICS_KEY].value), "automix": False, "tracks": False, "preset": None,
                           "requirements": [r for r in given.values() if r is not None]}
    am = given[AUTOMIX_KEY]
    if am is not None:
        v = am.value
        if isinstance(v, str) and v.strip().lower() not in ("true", "false", "yes", "no", "on", "off", "1", "0"):
            name = v.strip()
            if not _ID_RE.match(name):
                raise EditRequirementError(f"{AUTOMIX_KEY} must be true, false or a preset name, got {v!r}")
            out["automix"], out["preset"] = True, name
        else:
            out["automix"] = _bool(v, AUTOMIX_KEY)
    if given[PRESET_KEY] is not None and out["preset"] is None:
        out["preset"] = str(given[PRESET_KEY].value)
    for key, field, rng in ((WINDOW_KEY, "window_ms", WINDOW_RANGE), (RATE_KEY, "max_gain_rate_db_per_s", RATE_RANGE), (HOLD_KEY, "hold_below_dbfs", HOLD_RANGE)):
        r = given[key]
        if r is None:
            continue
        if not out["automix"]:
            raise EditRequirementError(f"{key} is set but {AUTOMIX_KEY} is not: refusing an ambiguous request")
        if key == HOLD_KEY and isinstance(r.value, str) and r.value.strip().lower() in ("off", "none"):
            continue   # the preset's hold is switched off explicitly: the shares follow the inputs' own noise
        out[field] = float(_number(r.value, key, rng[0], rng[1]))
        out[field + "_source"] = r.source
    if given[TRACKS_KEY] is not None:
        out["tracks"] = _bool(given[TRACKS_KEY].value, TRACKS_KEY)
    if given[FLOOR_KEY] is not None:
        if not out["tracks"]:
            raise EditRequirementError(f"{FLOOR_KEY} is set but {TRACKS_KEY} is not: refusing an ambiguous request")
        out["floor_gain_db"] = float(_number(given[FLOOR_KEY].value, FLOOR_KEY, GAIN_RANGE[0], GAIN_RANGE[1]))
        out["floor_gain_db_source"] = given[FLOOR_KEY].source
    if given[FLOOR_LANGUAGE_KEY] is not None:
        if not out["tracks"]:
            raise EditRequirementError(f"{FLOOR_LANGUAGE_KEY} is set but {TRACKS_KEY} is not: refusing an ambiguous request")
        code = str(given[FLOOR_LANGUAGE_KEY].value).strip().lower()
        if not _LANG_RE.match(code):
            raise EditRequirementError(f"{FLOOR_LANGUAGE_KEY} must be an ISO 639-2 code (eng, jpn, ...), got {given[FLOOR_LANGUAGE_KEY].value!r}")
        out["floor_language"] = code
    if out["tracks"] and not out["automix"]:
        raise EditRequirementError(f"{TRACKS_KEY} puts the floor mix under each interpreter: it needs {AUTOMIX_KEY} (the floor mix)")
    mics = out["microphones"]
    if out["tracks"] and not any(x["role"] == "interp" for x in mics):
        raise EditRequirementError(f"{TRACKS_KEY}: no microphone has role interp (an interpreter), so there is no language track to write")
    if out["automix"]:
        mixed = [x for x in mics if x["role"] in MIXED_ROLES]
        if len({x.get("clip") for x in mics}) == 1 and not AUTOMIX_INPUTS[0] <= len(mixed) <= AUTOMIX_INPUTS[1]:
            raise EditRequirementError(f"{AUTOMIX_KEY}: the automix shares {AUTOMIX_INPUTS[0]}..{AUTOMIX_INPUTS[1]} floor / presenter microphones, {len(mixed)} are declared")
        if mixed and all(x.get("pinned") for x in mixed):
            raise EditRequirementError(f"{AUTOMIX_KEY}: every microphone is pinned; at least one must be shared (that is what the automix does)")
    return out


def check_keys(values: Dict[str, Any]) -> None:
    """The `--set` gate (service._check_edit_requirements): the microphone keys parse, refinements only with their switch."""
    reqs = {k: Requirement(key=k, value=v, provenance="USER", source="cli") for k, v in (values or {}).items() if k in REQUIREMENT_KEYS}
    if PRESET_KEY in reqs:
        raise EditRequirementError(f"{PRESET_KEY} is recorded by the plan from the profile; choose a preset with {AUTOMIX_KEY}=<name>")
    parse_requirements(reqs)


def requested(values: Dict[str, Any]) -> bool:
    return bool((values or {}).get(MICS_KEY) is not None)


# ---- presets (profile data) -------------------------------------------------------------------------------------------
def presets_of(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    p = (profile_data or {}).get(PRESETS_DATA_KEY)
    return p if isinstance(p, dict) else {}


def preset_requirements(reqs: List[Requirement], profile_name: str, profile_data: Dict[str, Any]) -> List[Requirement]:
    """ADR-134: the profile preset's values as PROFILE requirements, only when the automix is asked for and only for the values the request
    does not give. The preset name is the request's (`audio.automix=<name>`) or the profile's `automix_presets.default`. A preset the
    profile does not have adds only its name (the decision blocks on it with the presets there are)."""
    m = {r.key: r for r in reqs}
    am = m.get(AUTOMIX_KEY)
    if am is None or am.provenance == "DEFAULT":
        return []
    v = am.value
    if v is False or (isinstance(v, str) and v.strip().lower() in ("false", "no", "off", "0")):
        return []
    presets = presets_of(profile_data)
    named = isinstance(v, str) and v.strip().lower() not in ("true", "yes", "on", "1")
    name = v.strip() if named else presets.get(DEFAULT_PRESET_FIELD)
    if not isinstance(name, str) or not name:
        return []
    src = f"profile:{profile_name}:{PRESETS_DATA_KEY}.{name}"
    out = [Requirement(key=PRESET_KEY, value=name, provenance="PROFILE", source=src)]
    values = presets.get(name)
    if not isinstance(values, dict):
        return out
    tracks = m.get(TRACKS_KEY)
    want_tracks = tracks is not None and tracks.value in (True, "true", "yes", "on", "1")
    for field, key in PRESET_FIELDS.items():
        if key in m or field not in values:
            continue
        if key == FLOOR_KEY and not want_tracks:
            continue
        out.append(Requirement(key=key, value=values[field], provenance="PROFILE", source=src))
    return out


# ---- the clip a microphone belongs to -------------------------------------------------------------------------------
def clip_of(mic: Dict[str, Any], assets: List[Any]) -> Tuple[Optional[Any], Optional[str]]:
    """(asset, None) or (None, why): `clip` names an input by path or file name; without it, the only input with a picture."""
    videos = [a for a in assets if (a.technical or {}).get("video")]
    want = mic.get("clip")
    if want:
        hits = [a for a in assets if os.path.realpath(a.path) == os.path.realpath(os.path.abspath(want)) or os.path.basename(a.path) == os.path.basename(want)]
        if len(hits) != 1:
            return None, f"clip {want!r} names {'no input' if not hits else 'several inputs'}"
        return hits[0], None
    if len(videos) != 1:
        return None, (f"{len(videos)} inputs have a picture: name the clip of each microphone (clip: <input file>)" if videos else "no input has a picture to carry the microphones' mix")
    return videos[0], None


def sample_rate_of(technical: Dict[str, Any]) -> Optional[int]:
    a = (technical or {}).get("audio") or {}
    try:
        return int(a.get("sample_rate")) if a.get("sample_rate") else None
    except (TypeError, ValueError):
        return None


def channels_of(technical: Dict[str, Any]) -> Optional[int]:
    a = (technical or {}).get("audio") or {}
    try:
        return int(a.get("channels")) if a.get("channels") is not None else None
    except (TypeError, ValueError):
        return None


def declared(mic: Dict[str, Any], sha256: Optional[str] = None) -> Dict[str, Any]:
    """The IR record of a microphone on its asset (`assets[<id>].microphones[]`): who, what role, where its sound is."""
    rec: Dict[str, Any] = {k: mic[k] for k in ("id", "name", "speaker", "role", "language") if k in mic}
    if "path" in mic:
        rec["source"] = {"kind": "file", "path": mic["path"], **({"audio_stream": mic["audio_stream"]} if "audio_stream" in mic else {}),
                         **({"sha256": sha256} if sha256 else {})}
    else:
        rec["source"] = {"kind": "channel", "channel": mic["channel"]}
    for k in ("pinned", "weight_db", "gain_db", "pan"):
        if k in mic:
            rec[k] = mic[k]
    return rec


# ---- decisions ------------------------------------------------------------------------------------------------------
def decide_microphones(eng: Any, m: Dict[str, Requirement], analysis: Any, approval_for: Callable, cap_block: Callable, probe_ids_of: Callable,
                       channel_ok: Optional[bool], conflicts: Dict[str, Any]) -> None:
    """One `audio.automix` decision per clip with microphones (TRANSFORM, or BLOCK with the reason), and its `audio.language_tracks`.
    `channel_ok`: whether the installed engine's audio tool takes --channel / --sample-rate (None: not known here, checked at run time).
    `conflicts`: what else the request asks that the microphones' path cannot follow ({audio_production: bool, timing: [keys], inputs: n})."""
    req = parse_requirements(m)
    if req is None:
        return
    ev_req = [r.id for r in req["requirements"]]
    mics = req["microphones"]
    by_clip: Dict[str, List[Dict[str, Any]]] = {}
    for mic in mics:
        asset, why = clip_of(mic, analysis.assets)
        if asset is None:
            eng.decide(subject=SUBJECT, type="BLOCK", decision=f"BLOCK: microphone {mic['id']} has no clip", reason=why, confidence=1.0,
                       evidence=ev_req + probe_ids_of([a.id for a in analysis.assets]), risk="HIGH", approval="BLOCK", provenance="USER",
                       params={"microphone": mic["id"]}, requirements=req["requirements"])
            return
        by_clip.setdefault(asset.id, []).append(mic)
    assets = {a.id: a for a in analysis.assets}
    for aid, group in by_clip.items():
        a = assets[aid]
        ev = ev_req + probe_ids_of([aid])

        def block(why: str, subject: str = SUBJECT) -> None:
            eng.decide(subject=subject, type="BLOCK", decision=f"BLOCK: {'automix' if subject == SUBJECT else 'language tracks'} of {aid}'s microphones", reason=why,
                       confidence=1.0, evidence=ev, risk="HIGH", approval="BLOCK", provenance="USER", params={"asset_id": aid, "microphones": [x["id"] for x in group]},
                       requirements=req["requirements"])
        if not req["automix"]:
            eng.decide(subject=SUBJECT, type="KEEP", decision=f"microphones of {aid} recorded (no automix asked)",
                       reason="the microphones are declared on the clip (who speaks on which track); without audio.automix the clip's own sound is kept",
                       confidence=1.0, evidence=ev, risk="LOW", approval="AUTO", provenance="USER", params={"asset_id": aid, "microphones": [x["id"] for x in group]},
                       requirements=req["requirements"])
            continue
        if conflicts.get("audio_production"):
            block("audio.production delivers a file's audio alone; the microphones' automix is the sound of the clip's picture: ask for one of them")
            continue
        if not (a.technical or {}).get("video"):
            block(f"{os.path.basename(a.path)} has no picture: the microphones' mix replaces the sound of a clip (for audio files alone use audio.production with audio.mix)")
            continue
        sr = sample_rate_of(a.technical)
        if not sr:
            block(f"{os.path.basename(a.path)} has no audio stream whose sample rate was measured: AUTOMIX needs every microphone at one rate, the clip's")
            continue
        chans = channels_of(a.technical) or 0
        bad = [x for x in group if "channel" in x and x["channel"] >= chans]
        if bad:
            block(f"microphone(s) {', '.join(x['id'] for x in bad)} name channel(s) {', '.join(str(x['channel']) for x in bad)}, but the clip's audio has {chans} channel(s) (0..{max(chans - 1, 0)})")
            continue
        if channel_ok is False:
            block("the installed engine's audio tool does not take --channel / --sample-rate (the reference engine since 3.7.0): the microphones cannot be split to one file each")
            continue
        mixed = [x for x in group if x["role"] in MIXED_ROLES]
        interp = [x for x in group if x["role"] == "interp"]
        if not AUTOMIX_INPUTS[0] <= len(mixed) <= AUTOMIX_INPUTS[1]:
            block(f"the automix shares {AUTOMIX_INPUTS[0]}..{AUTOMIX_INPUTS[1]} floor / presenter microphones, {aid} has {len(mixed)}")
            continue
        missing = [k for k in ("window_ms", "max_gain_rate_db_per_s") if k not in req]
        if missing:
            presets = sorted(k for k in presets_names(m))
            block(f"no value for {', '.join(missing)}: audio-production's AUTOMIX has no default for them and "
                  + (f"the profile has no automix preset {req['preset']!r}" if req.get("preset") else "the profile names no default automix preset")
                  + (f" (presets: {', '.join(presets)})" if presets else "") + f"; give {', '.join(PRESET_FIELDS[k] for k in missing)} or a preset")
            continue
        channels = [{k: x[k] for k in ("gain_db", "weight_db", "pinned") if k in x} for x in mixed]
        params: Dict[str, Any] = {"asset_id": aid, "microphones": [{k: v for k, v in x.items() if k != "clip"} for x in group], "inputs": [x["id"] for x in mixed], "window_ms": req["window_ms"], "max_gain_rate_db_per_s": req["max_gain_rate_db_per_s"],
                                  "channels": channels, "sample_rate": sr,
                                  "values_from": {k: req.get(k + "_source") for k in ("window_ms", "max_gain_rate_db_per_s", "hold_below_dbfs") if k in req}}
        if "hold_below_dbfs" in req:
            params["hold_below_dbfs"] = req["hold_below_dbfs"]
        if req.get("preset"):
            params["preset"] = req["preset"]
        words = (f"window {req['window_ms']:g} ms, gain rate <= {req['max_gain_rate_db_per_s']:g} dB/s"
                 + (f", hold below {req['hold_below_dbfs']:g} dBFS" if "hold_below_dbfs" in req else "") + (f"; preset {req['preset']}" if req.get("preset") else ""))
        pinned = [x["id"] for x in mixed if x.get("pinned")]
        files = [x["id"] for x in mixed if "path" in x]
        eng.decide(subject=SUBJECT, type="TRANSFORM",
                   decision=f"automix {', '.join(x['id'] + ' (' + x['name'] + ')' for x in mixed)} → the sound of {aid} ({words})",
                   reason="user asked for the gain-sharing automix of the clip's microphones: the shared gains sum to 0 dB, so the mix has the noise of one open "
                          "microphone and the others are down by as much as their level is below the talker's; each microphone is first written as a mono WAV at the "
                          f"clip's {sr} Hz (the engine's audio tool" + (" --channel" if any("channel" in x for x in mixed) else "") + "), and the mix replaces the clip's sound "
                          "(picture stream-copied) before anything else edits it"
                          + (f"; {', '.join(pinned)} stay open (pinned)" if pinned else "")
                          + (f"; {', '.join(files)} are separate files assumed to start with the clip (no offset is measured)" if files else ""),
                   confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for(SUBJECT, explicit=m[AUTOMIX_KEY]), provenance="USER", params=params,
                   requirements=req["requirements"])
        for sk in (EXTRACT_SKILL, AUTOMIX_SKILL, "audio_mux"):
            cap_block(sk, f"capability.{sk}")
        if not req["tracks"]:
            continue
        why = None
        if not interp:
            why = f"{aid} has no interpreter microphone (role interp)"
        elif len(interp) > TRACKS_INPUTS[1] - 1:
            why = f"MIX_TRACKS takes the floor and at most {TRACKS_INPUTS[1] - 1} interpreter microphones, {aid} has {len(interp)}"
        elif conflicts.get("inputs", 1) > 1:
            why = "the language tracks follow one clip's timeline; this plan has several inputs (a programme of several clips is not followed)"
        elif conflicts.get("timing"):
            why = f"{', '.join(conflicts['timing'])} change(s) the clip's timeline after its trim, which the language tracks do not follow"
        elif conflicts.get("track_ok") is False:
            why = "the installed engine's audio tool does not write named, language-tagged tracks (--track, the reference engine 4.x), which MIX_TRACKS with languages needs"
        elif "floor_gain_db" not in req:
            why = f"no floor level under the interpreters: give {FLOOR_KEY} or a preset with floor_gain_db"
        if why:
            block(why, TRACKS_SUBJECT)
            continue
        # one track of the floor itself (at 0 dB, the floor microphones' language or `und`), then one per interpreter language (the floor at
        # floor_gain_db under its interpreters). MIX_TRACKS sets one level per input for every language, so the floor is an input per track; input 0
        # is a muted copy that sets the duration (a level of input 0 would apply to every language)
        floor_langs = sorted({x["language"] for x in mixed if x.get("language")})
        floor_lang = req.get("floor_language") or (floor_langs[0] if len(floor_langs) == 1 else "und")
        inputs: List[str] = ["floor", "floor"]
        levels: List[Dict[str, Any]] = [{"gain_db": 0.0, "mute": True}, {"gain_db": 0.0, "mute": False}]
        langs: List[Dict[str, Any]] = [{"language_id": "floor", "language": floor_lang, "title": "Floor", "inputs": [1]}]
        for code in dict.fromkeys(x["language"] for x in interp):
            mine = [x for x in interp if x["language"] == code]
            idx = [len(inputs)]
            inputs.append("floor")
            levels.append({"gain_db": req["floor_gain_db"], "mute": False})
            for x in mine:
                idx.append(len(inputs))
                inputs.append(x["id"])
                levels.append({"gain_db": float(x.get("gain_db", 0.0)), "mute": False, **({"pan": x["pan"]} if "pan" in x else {})})
            langs.append({"language_id": code if code != floor_lang else f"{code}_interp", "language": code, "title": " + ".join(x["name"] for x in mine)[:64], "inputs": idx})
        if len(inputs) > TRACKS_INPUTS[1]:
            block(f"the floor track and {len(langs) - 1} interpreter language(s) take {len(inputs)} MIX_TRACKS inputs (one floor copy per track); the Skill takes at most {TRACKS_INPUTS[1]}",
                  TRACKS_SUBJECT)
            continue
        eng.decide(subject=TRACKS_SUBJECT, type="TRANSFORM",
                   decision=f"language tracks of {aid}: floor ({floor_lang}), " + ", ".join(f"{l['language']} ({', '.join(inputs[i] for i in l['inputs'][1:])} over the floor at {req['floor_gain_db']:+g} dB)"
                                                                                        for l in langs[1:]),
                   reason="user asked for interpretation tracks: the floor mix as one track and each language as the floor under that language's interpreter(s), "
                          "each a language-tagged track of an MKV next to the programme (audio-production MIX_TRACKS); the programme's own sound stays the floor mix; "
                          "the tracks are cut with the clip's trim so they stay on the delivered timeline",
                   confidence=1.0, evidence=ev, risk="MEDIUM", approval=approval_for(TRACKS_SUBJECT, explicit=m[TRACKS_KEY]), provenance="USER",
                   params={"asset_id": aid, "inputs": inputs, "levels": levels, "languages": langs, "format": TRACKS_FORMAT, "floor_gain_db": req["floor_gain_db"],
                           "floor_language": floor_lang}, requirements=req["requirements"])
        cap_block(TRACKS_SKILL, f"capability.{TRACKS_SKILL}")


def presets_names(m: Dict[str, Requirement]) -> List[str]:
    """The presets the profile has, as recorded on the plan's requirements (none when the profile has none)."""
    r = m.get(PRESET_KEY)
    return [str(r.value)] if r is not None and r.value else []


# ---- transcription ---------------------------------------------------------------------------------------------------
def transcript_name(mic: Dict[str, Any]) -> str:
    """The speaker id sent to transcription-skill for one microphone: its name when the Skill takes it (letters, digits, space . _ -),
    else its id. The display spelling (name / speaker / the entity memory's) is recorded beside it."""
    for c in (mic.get("speaker"), mic.get("name")):
        if isinstance(c, str) and TRANSCRIPT_NAME_RE.match(c):
            return c
    return mic["id"]


def transcription_microphones(group: List[Dict[str, Any]], files: Dict[str, str]) -> List[Dict[str, Any]]:
    """transcription/microphones' `microphones` input for the floor / presenter microphones of a clip: {input, name, audio_stream?}.
    `files`: mic id → the WAV the analysis wrote for a channel microphone (a file microphone is read as it is)."""
    out: List[Dict[str, Any]] = []
    for x in group:
        if x["role"] not in MIXED_ROLES:
            continue
        item: Dict[str, Any] = {"input": x["path"] if "path" in x else files[x["id"]], "name": transcript_name(x)}
        if "path" in x and x.get("audio_stream") is not None:
            item["audio_stream"] = x["audio_stream"]
        out.append(item)
    return out


def speakers_of(group: List[Dict[str, Any]]) -> Dict[str, str]:
    """speaker id in the transcript → the spelling shown for it (the entity memory's canonical name, the microphone's speaker, else its name)."""
    return {transcript_name(x): str(x.get("speaker") or x["name"]) for x in group if x["role"] in MIXED_ROLES}


def overlapping_of(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The transcript's `overlapping_segments` (words said while someone else was talking), compactly: what the plan keeps of them."""
    out = []
    for s in (transcript or {}).get("overlapping_segments") or []:
        if isinstance(s, dict):
            out.append({k: s.get(k) for k in ("id", "start", "end", "speaker_id", "text")})
    return out


def decide_transcript(eng: Any, analysis: Any, mics_by_asset: Dict[str, List[Dict[str, Any]]]) -> None:
    """A KEEP record per clip transcribed by microphone: the speakers, and the overlapping speech kept beside the segments (never dropped)."""
    for aid, group in mics_by_asset.items():
        tr = next((o for o in analysis.observations if o.kind == "transcript" and o.asset_id == aid and isinstance(o.data, dict) and o.data.get("microphone_merge")), None)
        if tr is None:
            continue
        over = overlapping_of(tr.data)
        counts = dict((tr.data.get("microphone_merge") or {}).get("counts") or {})
        eng.decide(subject=TRANSCRIPT_SUBJECT, type="KEEP",
                   decision=f"transcript of {aid} by microphone: {len(speakers_of(group))} speaker(s), {len(over)} overlapping stretch(es) kept",
                   reason="each microphone was transcribed on its own (transcription/microphones), the microphone's name is the speaker; words said while "
                          "someone else was talking are in overlapping_segments, kept here so they are not lost (subtitles are made from the segments)",
                   confidence=1.0, evidence=[tr.id], risk="LOW", approval="AUTO", provenance="SYSTEM",
                   params={"asset_id": aid, "transcript_id": tr.id, "speakers": speakers_of(group), "overlapping_segments": over,
                           "counts": {k: counts[k] for k in ("segments", "words", "words_dropped_as_crosstalk", "overlapping_segments", "overlapping_words") if k in counts}})


# ---- IR -------------------------------------------------------------------------------------------------------------
def mic_output(asset_id: str, mic_id: str) -> str:
    return f"{asset_id}_mic_{mic_id}"


def ir_automix(asset_id: str, dec: Any, mics: List[Dict[str, Any]], video_input: str) -> Dict[str, Any]:
    """The IR record of one clip's automix (`audio.automix[]`): the microphones in input order, each with the WAV it is written as, the
    AUTOMIX parameters, and the mux of the mix under the clip's picture. The language tracks are added by the planner after the trim."""
    p = dec.params
    by_id = {x["id"]: x for x in mics}
    used = list(p["inputs"])
    rec: Dict[str, Any] = {"type": SUBJECT, "asset": asset_id, "sample_rate": int(p["sample_rate"]),
                           "microphones": [{"id": i, "output": mic_output(asset_id, i), **({"channel": by_id[i]["channel"]} if "channel" in by_id[i] else {"path": by_id[i]["path"]}),
                                            **({"audio_stream": by_id[i]["audio_stream"]} if "audio_stream" in by_id[i] else {})} for i in used],
                           "window_ms": p["window_ms"], "max_gain_rate_db_per_s": p["max_gain_rate_db_per_s"], "channels": [dict(c) for c in p["channels"]],
                           "output": f"{asset_id}_automix", "mux": {"video": video_input, "audio": f"{asset_id}_automix", "output": f"{asset_id}_mic_mix"},
                           "decision_ids": [dec.id]}
    if p.get("hold_below_dbfs") is not None:
        rec["hold_below_dbfs"] = p["hold_below_dbfs"]
    if p.get("preset"):
        rec["preset"] = p["preset"]
    return rec


def ir_tracks(asset_id: str, dec: Any, mics: List[Dict[str, Any]], floor: str, remove: List[List[float]]) -> Dict[str, Any]:
    """`audio.automix[].language_tracks`: MIX_TRACKS over the floor mix (input 0 muted, setting the duration; one copy per track) and the
    interpreter microphones, each distinct input cut like the picture first when the clip is trimmed, as one MKV of language-tagged tracks."""
    p = dec.params
    by_id = {x["id"]: x for x in mics}
    interp = list(dict.fromkeys(i for i in p["inputs"] if i != "floor"))
    extra = [{"id": i, "output": mic_output(asset_id, i), **({"channel": by_id[i]["channel"]} if "channel" in by_id[i] else {"path": by_id[i]["path"]}),
              **({"audio_stream": by_id[i]["audio_stream"]} if "audio_stream" in by_id[i] else {})} for i in interp]
    ins = [floor if i == "floor" else mic_output(asset_id, i) for i in p["inputs"]]
    rec: Dict[str, Any] = {"microphones": extra, "inputs": ins, "levels": [dict(lv) for lv in p["levels"]], "languages": [dict(l, inputs=list(l["inputs"])) for l in p["languages"]],
                           "format": TRACKS_FORMAT, "output": f"{asset_id}_language_tracks", "decision_ids": [dec.id]}
    if remove:
        rec["remove"] = [list(r) for r in remove]
        rec["cut"] = [f"{i}_cut" for i in ins]
    return rec


def automix_problems(rec: Dict[str, Any], asset: Optional[Dict[str, Any]], allowed: List[str]) -> List[str]:
    """Validator checks of one `audio.automix[]` record against its asset's declared microphones (`assets[<id>].microphones`)."""
    errs: List[str] = []
    aid = rec.get("asset")
    if asset is None:
        return [f"audio.automix: asset {aid!r} is not in the IR"]
    decl = {x.get("id"): x for x in asset.get("microphones") or []}
    if not decl:
        return [f"audio.automix {aid}: the asset declares no microphones"]
    mics = rec.get("microphones") or []
    if not AUTOMIX_INPUTS[0] <= len(mics) <= AUTOMIX_INPUTS[1]:
        errs.append(f"audio.automix {aid}: AUTOMIX takes {AUTOMIX_INPUTS[0]}..{AUTOMIX_INPUTS[1]} microphones, got {len(mics)}")
    chans = channels_of(asset.get("technical") or {}) or 0

    def check_mic(x: Dict[str, Any], roles: Tuple[str, ...]) -> None:
        d = decl.get(x.get("id"))
        if d is None:
            errs.append(f"audio.automix {aid}: microphone {x.get('id')!r} is not declared on the asset")
            return
        if d.get("role") not in roles:
            errs.append(f"audio.automix {aid}: microphone {x['id']} has role {d.get('role')}, not one of {roles}")
        src = d.get("source") or {}
        if src.get("kind") == "channel":
            if x.get("channel") != src.get("channel") or "path" in x:
                errs.append(f"audio.automix {aid}: microphone {x['id']} is channel {src.get('channel')} of the clip")
            elif not 0 <= int(src.get("channel", -1)) < chans:
                errs.append(f"audio.automix {aid}: channel {src.get('channel')} is not one of the clip's {chans}")
        else:
            if x.get("path") != src.get("path") or "channel" in x:
                errs.append(f"audio.automix {aid}: microphone {x['id']} is the file {src.get('path')}")
            elif not any(os.path.realpath(str(src.get("path"))).startswith(os.path.realpath(r) + os.sep) for r in allowed):
                errs.append(f"audio.automix {aid}: microphone file {src.get('path')} is outside the allowed input roots")
        if x.get("output") != mic_output(aid, x.get("id", "")):
            errs.append(f"audio.automix {aid}: microphone {x.get('id')} is written as {mic_output(aid, x.get('id', ''))}")
    for x in mics:
        check_mic(x, MIXED_ROLES)
    for k, rng in (("window_ms", WINDOW_RANGE), ("max_gain_rate_db_per_s", RATE_RANGE)):
        v = rec.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not rng[0] <= float(v) <= rng[1]:
            errs.append(f"audio.automix {aid}: {k} must be {rng[0]:g}..{rng[1]:g} (no default), got {v!r}")
    if rec.get("hold_below_dbfs") is not None and not HOLD_RANGE[0] <= float(rec["hold_below_dbfs"]) <= HOLD_RANGE[1]:
        errs.append(f"audio.automix {aid}: hold_below_dbfs out of range")
    ch = rec.get("channels") or []
    if len(ch) != len(mics):
        errs.append(f"audio.automix {aid}: one channels entry per microphone ({len(mics)}), got {len(ch)}")
    elif ch and all(c.get("pinned") for c in ch):
        errs.append(f"audio.automix {aid}: at least one microphone must be shared (not pinned)")
    if (rec.get("mux") or {}).get("audio") != rec.get("output"):
        errs.append(f"audio.automix {aid}: the mux must put the automix output under the picture")
    tr = rec.get("language_tracks")
    if tr:
        for x in tr.get("microphones") or []:
            check_mic(x, ("interp",))
        ins = list(tr.get("inputs") or [])
        n = len(ins)
        if not TRACKS_INPUTS[0] <= n <= TRACKS_INPUTS[1] or (ins or [None])[0] != rec.get("output"):
            errs.append(f"audio.automix {aid}: the language tracks take the floor mix first and at most {TRACKS_INPUTS[1]} inputs")
        lv = list(tr.get("levels") or [])
        if len(lv) != n:
            errs.append(f"audio.automix {aid}: one level per language-track input")
        elif not lv[0].get("mute"):
            errs.append(f"audio.automix {aid}: input 0 of the language tracks is the muted floor copy that sets the duration")
        named = [k for l in tr.get("languages") or [] for k in l.get("inputs") or []]
        if sorted(named) != list(range(1, n)) or not all(_LANG_RE.match(str(l.get("language") or "")) for l in tr.get("languages") or []):
            errs.append(f"audio.automix {aid}: every input after the first in exactly one language (ISO 639-2)")
        interp_out = {x.get("output") for x in tr.get("microphones") or []}
        if any(i not in interp_out and i != rec.get("output") for i in ins):
            errs.append(f"audio.automix {aid}: a language-track input is neither the floor mix nor an interpreter microphone")
        if tr.get("format") != TRACKS_FORMAT:
            errs.append(f"audio.automix {aid}: the language tracks are written as {TRACKS_FORMAT}")
        if tr.get("remove") is not None and len(tr.get("cut") or []) != n:
            errs.append(f"audio.automix {aid}: every language-track input is cut like the picture")
    return errs
