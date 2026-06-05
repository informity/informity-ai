import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from informity import config
from informity.api import routes_settings
from informity.api.schemas import CurrentChatUpdateRequest, SettingsUpdateRequest


@pytest.mark.asyncio
async def test_get_settings_normalizes_without_mutating_config_singleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])
    monkeypatch.setattr(routes_settings, '_visible_role_ids', lambda: {'assistant', 'researcher'})
    monkeypatch.setattr(config.settings, 'llm_model_filename', 'Qwen_Qwen3.5-9B-Q4_K_M.gguf   ')
    monkeypatch.setattr(config.settings, 'llm_model_id', '')
    monkeypatch.setattr(config.settings, 'enabled_chat_role_ids', ['assistant', 'invalid', 'researcher'])
    monkeypatch.setattr(config.settings, 'enable_chat_roles', False)

    response = await routes_settings.get_settings()

    assert response.llm_model_filename == 'Qwen_Qwen3.5-9B-Q4_K_M.gguf'
    assert response.enabled_chat_role_ids == ['assistant', 'researcher']
    assert response.enable_chat_roles is True
    assert config.settings.llm_model_filename == 'Qwen_Qwen3.5-9B-Q4_K_M.gguf   '
    assert config.settings.llm_model_id == ''
    assert config.settings.enabled_chat_role_ids == ['assistant', 'invalid', 'researcher']
    assert config.settings.enable_chat_roles is False


@pytest.mark.asyncio
async def test_mcp_stdio_start_does_not_mark_manager_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop_emit_log_event(**kwargs) -> None:
        _ = kwargs

    monkeypatch.setattr(config.settings, 'mcp_enabled', True)
    monkeypatch.setattr(config.settings, 'mcp_transport', 'stdio')
    monkeypatch.setattr(config.settings, 'mcp_http_host', '127.0.0.1')
    monkeypatch.setattr(config.settings, 'mcp_http_port', 8765)
    monkeypatch.setattr(config.settings, 'mcp_scope_mode', 'metadata_only')
    monkeypatch.setattr(routes_settings.mcp_lifecycle, '_running', False)
    monkeypatch.setattr(routes_settings.mcp_lifecycle, '_last_error', 'previous-error')
    monkeypatch.setattr('informity.mcp.lifecycle.emit_log_event', _noop_emit_log_event)

    await routes_settings.mcp_lifecycle.start_from_settings()

    assert routes_settings.mcp_lifecycle.running is False
    assert routes_settings.mcp_lifecycle.last_error is None


@pytest.mark.asyncio
async def test_concurrent_settings_and_current_chat_updates_keep_valid_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    updates = []
    for i in range(10):
        updates.append(routes_settings.update_settings(SettingsUpdateRequest(log_level='info')))
        updates.append(
            routes_settings.update_current_chat(
                CurrentChatUpdateRequest(current_chat_id=f'chat-{i}'),
            ),
        )

    await asyncio.gather(*updates)

    config_path = tmp_path / 'config.json'
    assert config_path.exists()

    payload = json.loads(config_path.read_text(encoding='utf-8'))
    assert payload['log_level'] == 'info'
    assert payload['current_chat_id'].startswith('chat-')


@pytest.mark.asyncio
async def test_unknown_settings_field_is_ignored_and_treated_as_empty_update_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    # Unknown keys are stripped by Pydantic; empty payload is a no-op (200), not 400.
    response = await routes_settings.update_settings(
        SettingsUpdateRequest.model_validate({'legacy_field': 'value'}),
    )
    assert response is not None

    config_path = tmp_path / 'config.json'
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding='utf-8'))
        assert 'legacy_field' not in payload


@pytest.mark.asyncio
async def test_scan_file_timeout_seconds_rejects_out_of_range_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    with pytest.raises(HTTPException) as exc_info_low:
        await routes_settings.update_settings(SettingsUpdateRequest(scan_file_timeout_seconds=-1))
    assert exc_info_low.value.status_code == 400
    assert 'scan_file_timeout_seconds must be between 1 and 600' in str(exc_info_low.value.detail)

    with pytest.raises(HTTPException) as exc_info_high:
        await routes_settings.update_settings(SettingsUpdateRequest(scan_file_timeout_seconds=601))
    assert exc_info_high.value.status_code == 400
    assert 'scan_file_timeout_seconds must be between 1 and 600' in str(exc_info_high.value.detail)


@pytest.mark.asyncio
async def test_scan_file_timeout_seconds_rejects_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    with pytest.raises(HTTPException) as exc_info:
        await routes_settings.update_settings(SettingsUpdateRequest(scan_file_timeout_seconds=0))
    assert exc_info.value.status_code == 400
    assert 'scan_file_timeout_seconds must be between 1 and 600' in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_scan_file_timeout_seconds_updates_runtime_policy_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(scan_file_timeout_seconds=550)
    )
    assert updated.scan_file_timeout_seconds == 550
    assert config.settings.scan_timeout_policy.default.max_seconds == 550
    assert config.settings.scan_timeout_policy.overrides['filesystem:file'].max_seconds == 550


@pytest.mark.asyncio
async def test_web_search_provider_settings_support_dual_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(
            tavily_api_key='  tvly-test  ',
            linkup_api_key='  lk-test  ',
            web_search_primary_provider='linkup',
        ),
    )

    assert updated.tavily_api_key_set is True
    assert updated.linkup_api_key_set is True
    assert updated.web_search_configured is True
    assert updated.web_search_primary_provider == 'linkup'
    assert config.settings.tavily_api_key == 'tvly-test'
    assert config.settings.linkup_api_key == 'lk-test'


@pytest.mark.asyncio
async def test_translate_defaults_round_trip_through_settings_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(
            translate_default_language='German',
            translate_default_tone='formal',
            translate_pinned_languages=['French', 'German', 'French'],
        ),
    )

    assert updated.translate_default_language == 'German'
    assert updated.translate_default_tone == 'formal'
    assert updated.translate_pinned_languages == ['French', 'German']
    assert config.settings.translate_default_language == 'German'
    assert config.settings.translate_default_tone == 'formal'
    assert config.settings.translate_pinned_languages == ['French', 'German']

    reloaded = await routes_settings.get_settings()
    assert reloaded.translate_default_language == 'German'
    assert reloaded.translate_default_tone == 'formal'
    assert reloaded.translate_pinned_languages == ['French', 'German']


@pytest.mark.asyncio
async def test_translate_pinned_language_limit_round_trip_through_settings_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(
            translate_pinned_languages_limit=3,
            translate_pinned_languages=['French', 'German', 'Spanish', 'Italian'],
        ),
    )

    assert updated.translate_pinned_languages_limit == 3
    assert updated.translate_pinned_languages == ['French', 'German', 'Italian']
    assert config.settings.translate_pinned_languages_limit == 3
    assert config.settings.translate_pinned_languages == ['French', 'German', 'Italian']

    reloaded = await routes_settings.get_settings()
    assert reloaded.translate_pinned_languages_limit == 3
    assert reloaded.translate_pinned_languages == ['French', 'German', 'Italian']


@pytest.mark.asyncio
async def test_mcp_http_host_rejects_non_loopback_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    with pytest.raises(HTTPException) as exc_info:
        await routes_settings.update_settings(SettingsUpdateRequest(mcp_http_host='192.168.1.10'))
    assert exc_info.value.status_code == 400
    assert 'mcp_http_host must be loopback only' in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_mcp_settings_update_restarts_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    calls: list[str] = []

    async def _fake_restart() -> None:
        calls.append('restart')

    monkeypatch.setattr(routes_settings.mcp_lifecycle, 'restart_from_settings', _fake_restart)

    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(
            mcp_enabled=True,
            mcp_auto_start=False,
            mcp_transport='stdio',
            mcp_scope_mode='metadata_only',
        ),
    )

    assert updated.mcp_enabled is True
    assert updated.mcp_auto_start is True
    assert calls == ['restart']


@pytest.mark.asyncio
async def test_mcp_disabling_clears_access_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    await routes_settings.update_settings(
        SettingsUpdateRequest(
            mcp_enabled=True,
            mcp_transport='http',
            mcp_access_token='imcp_12345678901234567890123456789012',
        ),
    )
    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(
            mcp_enabled=False,
        ),
    )

    assert updated.mcp_enabled is False
    assert updated.mcp_access_token == ''

    payload = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
    assert payload.get('mcp_access_token', None) == ''


@pytest.mark.asyncio
async def test_mcp_switching_to_stdio_clears_access_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    await routes_settings.update_settings(
        SettingsUpdateRequest(
            mcp_enabled=True,
            mcp_transport='http',
            mcp_access_token='imcp_abcdefghijklmnopqrstuvwxyz123456',
        ),
    )
    updated = await routes_settings.update_settings(
        SettingsUpdateRequest(
            mcp_transport='stdio',
        ),
    )

    assert updated.mcp_transport == 'stdio'
    assert updated.mcp_access_token == ''

    payload = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
    assert payload.get('mcp_access_token', None) == ''
