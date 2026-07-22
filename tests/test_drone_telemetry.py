import pytest

from app.drone_telemetry import SrtParseError, parse_srt

DJI_SRT_TWO_SAMPLES = """1
00:00:00,000 --> 00:00:00,033
<font size="36">SrtCnt : 1, DiffTime : 33ms
2024-01-01 12:00:00
[iso : 100] [shutter : 1/1000.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 22.543000] [longitude: 114.057000] [rel_alt: 50.100 abs_alt: 120.400] </font>

2
00:00:00,033 --> 00:00:00,066
<font size="36">SrtCnt : 2, DiffTime : 33ms
2024-01-01 12:00:00
[iso : 100] [shutter : 1/1000.0] [fnum : 280] [ev : 0] [ct : 5500] [color_md : default] [focal_len : 24.00] [latitude: 22.543010] [longitude: 114.057010] [rel_alt: 51.200 abs_alt: 121.500] </font>

"""

PLAIN_SUBTITLE_NO_TELEMETRY = """1
00:00:00,000 --> 00:00:02,000
Just a regular subtitle, no drone data here.

"""


def test_parse_srt_missing_file_raises(tmp_path):
    with pytest.raises(SrtParseError, match="not found"):
        parse_srt(tmp_path / "missing.srt")


def test_parse_srt_extracts_altitude_and_gps(tmp_path):
    srt_path = tmp_path / "flight.srt"
    srt_path.write_text(DJI_SRT_TWO_SAMPLES, encoding="utf-8")

    telemetry = parse_srt(srt_path)

    assert telemetry.sample_count == 2
    assert telemetry.has_gps is True
    assert telemetry.min_altitude_m == 50.1
    assert telemetry.max_altitude_m == 51.2
    assert telemetry.avg_altitude_m == pytest.approx(50.65)


def test_parse_srt_plain_subtitle_has_no_telemetry(tmp_path):
    srt_path = tmp_path / "captions.srt"
    srt_path.write_text(PLAIN_SUBTITLE_NO_TELEMETRY, encoding="utf-8")

    telemetry = parse_srt(srt_path)

    assert telemetry.sample_count == 0
    assert telemetry.has_gps is False
    assert telemetry.avg_altitude_m is None
    assert telemetry.min_altitude_m is None
    assert telemetry.max_altitude_m is None
