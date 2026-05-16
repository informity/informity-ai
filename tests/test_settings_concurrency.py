import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from informity import config
from informity.api import routes_settings
from informity.api.schemas import CurrentChatUpdateRequest, SettingsUpdateRequest


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
async def test_unknown_settings_field_is_ignored_and_rejected_as_empty_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    with pytest.raises(HTTPException) as exc_info:
        await routes_settings.update_settings(SettingsUpdateRequest.model_validate({'legacy_field': 'value'}))
    assert exc_info.value.status_code == 400
    assert 'No fields provided to update' in str(exc_info.value.detail)


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
