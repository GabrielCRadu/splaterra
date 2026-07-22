from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class SrtParseError(Exception):
    pass


@dataclass
class DroneTelemetry:
    sample_count: int
    has_gps: bool
    avg_altitude_m: float | None
    min_altitude_m: float | None
    max_altitude_m: float | None


_ALTITUDE_RE = re.compile(r"rel_alt:\s*(-?\d+(?:\.\d+)?)")
_GPS_RE = re.compile(r"latitude:\s*(-?\d+(?:\.\d+)?)")


def parse_srt(srt_path: Path) -> DroneTelemetry:
    if not srt_path.is_file():
        raise SrtParseError(f"SRT file not found: {srt_path}")

    text = srt_path.read_text(encoding="utf-8", errors="ignore")

    altitudes = [float(match) for match in _ALTITUDE_RE.findall(text)]
    has_gps = _GPS_RE.search(text) is not None

    if not altitudes:
        return DroneTelemetry(
            sample_count=0,
            has_gps=has_gps,
            avg_altitude_m=None,
            min_altitude_m=None,
            max_altitude_m=None,
        )

    return DroneTelemetry(
        sample_count=len(altitudes),
        has_gps=has_gps,
        avg_altitude_m=sum(altitudes) / len(altitudes),
        min_altitude_m=min(altitudes),
        max_altitude_m=max(altitudes),
    )
