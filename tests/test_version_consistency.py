"""Test module for tests test version consistency."""

from typing import Any, cast

from informity.api.schemas import DiagnosticsResponse, HealthResponse
from informity.version import APP_VERSION


def _get_field_default(model_cls: type[Any], field_name: str) -> object:
    """Return a Pydantic field default safely."""
    fields = cast(dict[str, Any], model_cls.model_fields)
    return fields[field_name].default


def test_schema_version_defaults_use_app_version_constant() -> None:
    """Test schema version defaults use app version constant."""
    assert _get_field_default(HealthResponse, "version") == APP_VERSION
    assert _get_field_default(DiagnosticsResponse, "app_version") == APP_VERSION
