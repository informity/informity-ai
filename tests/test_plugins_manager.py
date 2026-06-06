from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from informity.plugins.manager import (
    PLUGIN_MANIFEST_FILENAME,
    PluginManifest,
    discover_plugins,
    install_plugin_from_path,
    load_plugin_state,
    save_plugin_state,
    set_global_plugins_enabled,
    set_plugin_enabled,
)


def _write_manifest(plugin_dir: Path, payload: dict[str, object]) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / PLUGIN_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding='utf-8')


def test_plugin_manifest_requires_hook_permissions() -> None:
    with pytest.raises(ValidationError):
        PluginManifest.model_validate(
            {
                'id': 'legal',
                'name': 'Legal',
                'version': '1.0.0',
                'plugin_type': 'role',
                'hook_points': ['output_formatting'],
                'permissions': ['prompt:append'],
            }
        )


def test_discover_plugins_reports_enabled_and_invalid_plugins(tmp_path: Path) -> None:
    app_data_dir = tmp_path / 'app-data'
    plugin_root = app_data_dir / 'plugins'
    plugin_root.mkdir(parents=True, exist_ok=True)

    _write_manifest(
        plugin_root / 'legal',
        {
            'id': 'legal',
            'name': 'Legal Lens',
            'version': '1.0.0',
            'plugin_type': 'role',
            'hook_points': ['system_prompt_fragment', 'retrieval_hint'],
            'permissions': ['prompt:append', 'retrieval:hint'],
            'capabilities': ['chat', 'retrieval'],
        },
    )
    (plugin_root / 'broken').mkdir()

    save_plugin_state(
        load_plugin_state(app_data_dir).model_copy(update={'enabled_plugin_ids': ['legal']}),
        app_data_dir,
    )

    inventory = discover_plugins(app_data_dir)

    assert inventory.total_plugins == 2
    assert inventory.enabled_plugins == 1
    assert inventory.invalid_plugins == 1
    legal = next(item for item in inventory.plugins if item.id == 'legal')
    broken = next(item for item in inventory.plugins if item.id == 'broken')
    assert legal.enabled is True
    assert legal.valid is True
    assert broken.valid is False
    assert broken.enabled is False


def test_install_toggle_and_global_kill_switch(tmp_path: Path) -> None:
    app_data_dir = tmp_path / 'app-data'
    source_dir = tmp_path / 'source-legal'
    _write_manifest(
        source_dir,
        {
            'id': 'legal',
            'name': 'Legal Lens',
            'version': '1.0.0',
            'plugin_type': 'role',
            'hook_points': ['system_prompt_fragment', 'retrieval_hint'],
            'permissions': ['prompt:append', 'retrieval:hint'],
            'capabilities': ['chat', 'retrieval'],
        },
    )
    (source_dir / 'readme.txt').write_text('plugin asset', encoding='utf-8')

    installed = install_plugin_from_path(source_dir, app_data_dir)
    assert installed.total_plugins == 1
    assert (app_data_dir / 'plugins' / 'legal' / 'readme.txt').exists()

    enabled_inventory = set_plugin_enabled('legal', True, app_data_dir)
    legal = next(item for item in enabled_inventory.plugins if item.id == 'legal')
    assert legal.enabled is True

    disabled_inventory = set_global_plugins_enabled(False, app_data_dir)
    legal_after_kill_switch = next(item for item in disabled_inventory.plugins if item.id == 'legal')
    assert disabled_inventory.global_enabled is False
    assert legal_after_kill_switch.enabled is False
