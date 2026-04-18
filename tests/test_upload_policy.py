from __future__ import annotations

from informity.upload_policy import is_allowed_mime


def test_is_allowed_mime_accepts_office_openxml_variants() -> None:
    assert is_allowed_mime('application/vnd.openxmlformats-officedocument.wordprocessingml.document') is True
    assert is_allowed_mime('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') is True
    assert is_allowed_mime('application/vnd.openxmlformats-officedocument.presentationml.presentation') is True


def test_is_allowed_mime_ignores_charset_parameters() -> None:
    assert is_allowed_mime('text/plain; charset=utf-8') is True
    assert is_allowed_mime('application/json; charset=UTF-8') is True


def test_is_allowed_mime_rejects_non_allowlisted_types() -> None:
    assert is_allowed_mime('image/png') is False
    assert is_allowed_mime('application/x-msdownload') is False

