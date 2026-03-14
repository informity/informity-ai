from informity.api.schemas import DiagnosticsResponse, HealthResponse
from informity.version import APP_VERSION


def test_schema_version_defaults_use_app_version_constant() -> None:
    assert HealthResponse.model_fields['version'].default == APP_VERSION
    assert DiagnosticsResponse.model_fields['app_version'].default == APP_VERSION
