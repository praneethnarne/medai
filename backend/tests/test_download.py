"""The completeness check must reject truncated files.

ISIC's manifest reports a size a constant 3,209 bytes below what S3 serves, so
size comparison is useless here and JPEG markers are what we rely on. If that
check regresses, a half-downloaded image silently enters the training set.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from data.download import looks_complete  # noqa: E402

SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def test_accepts_a_whole_jpeg(tmp_path):
    p = tmp_path / "whole.jpg"
    p.write_bytes(SOI + b"payload" * 50 + EOI)
    assert looks_complete(p)


def test_rejects_a_truncated_jpeg(tmp_path):
    p = tmp_path / "cut.jpg"
    p.write_bytes(SOI + b"payload" * 50)  # no end-of-image marker
    assert not looks_complete(p)


def test_rejects_an_empty_file(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    assert not looks_complete(p)


def test_rejects_a_non_jpeg(tmp_path):
    p = tmp_path / "html.jpg"
    p.write_bytes(b"<html>503 Service Unavailable</html>")
    assert not looks_complete(p)


def test_rejects_a_missing_file(tmp_path):
    assert not looks_complete(tmp_path / "nope.jpg")
