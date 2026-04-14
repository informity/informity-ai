from informity.api.env_vars_metadata import get_env_vars_response
from informity.config import Settings


def test_env_vars_metadata_covers_all_settings_fields() -> None:
    response = get_env_vars_response(Settings())
    env_names = {
        variable.name
        for group in response.groups
        for variable in group.variables
    }

    expected_setting_env_names = {
        f'INFORMITY_{field.upper()}'
        for field in Settings.model_fields
    }

    missing = sorted(expected_setting_env_names - env_names)
    assert not missing, f'Missing Settings env vars in metadata response: {missing}'


def test_env_vars_metadata_includes_runtime_environment_group() -> None:
    response = get_env_vars_response(Settings())
    runtime_group = next((group for group in response.groups if group.title == 'Runtime Environment'), None)
    assert runtime_group is not None
    runtime_names = {variable.name for variable in runtime_group.variables}
    assert 'INFORMITY_TAURI_SESSION_TOKEN' in runtime_names


def test_env_vars_metadata_redacts_runtime_secrets(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv('INFORMITY_TAURI_SESSION_TOKEN', 'top-secret-token')
    response = get_env_vars_response(Settings())
    runtime_group = next((group for group in response.groups if group.title == 'Runtime Environment'), None)
    assert runtime_group is not None
    token_item = next((item for item in runtime_group.variables if item.name == 'INFORMITY_TAURI_SESSION_TOKEN'), None)
    assert token_item is not None
    assert token_item.current_value == '***set***'
