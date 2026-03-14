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
async def test_default_response_mode_rejects_mode_not_supported_by_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(config.settings, 'llm_model_filename', 'Qwen3-14B-Q5_K_M.gguf')
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: ['Qwen3-14B-Q5_K_M.gguf'])

    with pytest.raises(HTTPException) as exc_info:
        await routes_settings.update_settings(SettingsUpdateRequest(default_response_mode='research'))
    assert exc_info.value.status_code == 400
    assert 'not supported by active model' in str(exc_info.value.detail)


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
    assert 'scan_file_timeout_seconds must be between 0 and 600' in str(exc_info_low.value.detail)

    with pytest.raises(HTTPException) as exc_info_high:
        await routes_settings.update_settings(SettingsUpdateRequest(scan_file_timeout_seconds=601))
    assert exc_info_high.value.status_code == 400
    assert 'scan_file_timeout_seconds must be between 0 and 600' in str(exc_info_high.value.detail)


@pytest.mark.asyncio
async def test_scan_file_timeout_seconds_accepts_zero_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_settings, '_list_available_models', lambda: [])

    updated = await routes_settings.update_settings(SettingsUpdateRequest(scan_file_timeout_seconds=0))
    assert updated.scan_file_timeout_seconds == 0
