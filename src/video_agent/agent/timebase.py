"""Frame rates and frame arithmetic by the ecosystem's timebase spec (velisa/timebase@1, ADR-120).

Every rate is an exact fraction of frames per second and every time an exact fraction of seconds. A rate written as a
rounded decimal (the media probe's `fps` has 3 decimals) is read by one table of NTSC names, never by
limit_denominator() (which reads 29.97 as 2997/100). video-editing-skill reads rates with the same functions (its ADR-040),
so the agent's planned concat timeline and the Skill's joins stay on one frame grid (ADR-115).
"""
import math
import re
from fractions import Fraction
from typing import Any

# tests/test_timebase.py runs the spec's conformance vectors (tests/fixtures/velisa/timebase-vectors.json) against these.
NTSC_BASES = (24, 30, 48, 60, 120, 240)
NTSC_TOLERANCE = Fraction(5, 1000)
MAX_RATE = Fraction(1000)
DOWN_TOLERANCE = Fraction(1, 1000)
_RATIO = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_DECIMAL = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


class RateError(ValueError):
    """A rate or time the timebase spec refuses. `code` is the spec's: INVALID_RATE or INVALID_TIME. Callers turn it into
    a refusal of their own."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _exact(value: Any, code: str, what: str) -> Fraction:
    if isinstance(value, bool):
        raise RateError(code, f"{what}: a boolean is not a number")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RateError(code, f"{what}: {value!r} is not finite")
        return Fraction(repr(value))
    if isinstance(value, str):
        m = _RATIO.match(value)
        if m:
            if int(m.group(2)) == 0:
                raise RateError(code, f"{what}: {value!r} has a zero denominator")
            return Fraction(int(m.group(1)), int(m.group(2)))
        d = _DECIMAL.match(value.strip())
        # a bounded text and exponent: Fraction("1e1000000000") would build a number of a billion digits before any range check
        if d and len(value.strip()) <= 64 and (not d.group(3) or abs(int(d.group(3)[1:])) <= 9):
            return Fraction(value.strip())
    raise RateError(code, f"{what}: {value!r} is not a number")


def round_half_up(x: Fraction) -> int:
    """The one rounding to the nearest integer: a half goes up, toward +infinity (floor(x + 1/2)). Python's round() takes a
    half to the even neighbour and is not used for frames or decimal text."""
    return math.floor(x + Fraction(1, 2))


def parse_rate(value: Any) -> Fraction:
    """A frame rate as an exact fraction. 'N/D' (or a Fraction) is taken as written, an integer is N/1, a decimal that is not
    an integer and lies within 0.005 of base*1000/1001 (base 24, 30, 48, 60, 120, 240) is that NTSC rate ('23.976', '23.98',
    '29.97', '59.94', ...), any other decimal is its exact value ('12.5' is 25/2). A float is read through its shortest
    decimal text (repr). The rate must be in (0, 1000]."""
    text: Any = value.strip() if isinstance(value, str) else value
    try:
        exact = _exact(text, "INVALID_RATE", "frame rate")
    except RateError:
        raise RateError("INVALID_RATE", f"frame rate {value!r} is not N/D, an integer or a decimal") from None
    as_ratio = isinstance(value, Fraction) or (isinstance(text, str) and "/" in text)
    rate = exact
    if not as_ratio and exact.denominator != 1:
        for base in NTSC_BASES:
            ntsc = Fraction(base * 1000, 1001)
            if abs(exact - ntsc) <= NTSC_TOLERANCE:
                rate = ntsc
                break
    if rate <= 0 or rate > MAX_RATE:
        raise RateError("INVALID_RATE", f"frame rate {value!r} is out of range (0, 1000]")
    return rate


def rate_str(rate: Any) -> str:
    """The canonical text of a rate: '25' or '30000/1001'."""
    r = parse_rate(rate)
    return str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"


def nominal(rate: Any) -> int:
    """Timecode frames per second: the rate rounded half up (30 for 29.97)."""
    return max(1, round_half_up(parse_rate(rate)))


def same_rate(a: Any, b: Any) -> bool:
    """Two rates are the same only when their exact fractions are equal (no float tolerance)."""
    return parse_rate(a) == parse_rate(b)


def probed_rate(rate: Any) -> Fraction:
    """A rate as the media probe reports it (`fps`, 3 decimals, a half up) and read back: what a rate measured on a file
    is compared with. 30000/1001 stays 30000/1001; a rate no 3-decimal text names exactly (7/3) becomes 2333/1000, which
    is also what the probe of a 7/3 file reads as, so the two compare equal."""
    return parse_rate(format_seconds(parse_rate(rate), 3))


def frames_at(seconds: Any, rate: Any, mode: str = "nearest", tolerance: Any = DOWN_TOLERANCE) -> int:
    """Frames of a time: `nearest` (floor(t*rate + 1/2), positions) or `down` (floor((t + 1 ms)*rate), joined lengths)."""
    t = _exact(seconds, "INVALID_TIME", "time")
    r = parse_rate(rate)
    if mode == "nearest":
        return round_half_up(t * r)
    if mode == "down":
        return math.floor((t + _exact(tolerance, "INVALID_TIME", "tolerance")) * r)
    raise RateError("INVALID_TIME", f"unknown rounding mode {mode!r} (nearest | down)")


def seconds_of(frames: int, rate: Any) -> Fraction:
    """The exact start time of frame `frames`: frames / rate."""
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise RateError("INVALID_TIME", f"frame count {frames!r} is not an integer")
    return Fraction(frames) / parse_rate(rate)


def transition_frames(seconds: Any, rate: Any) -> int:
    """A transition's length in whole frames: the nearest, a half up, at least one."""
    return max(1, frames_at(seconds, rate, "nearest"))


def format_seconds(seconds: Any, places: int = 6) -> str:
    """Decimal text of a time to `places` digits, the nearest, a half up."""
    t = _exact(seconds, "INVALID_TIME", "time")
    n = round_half_up(t * 10 ** places)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if places == 0:
        return f"{sign}{n}"
    return f"{sign}{n // 10 ** places}.{n % 10 ** places:0{places}d}"
