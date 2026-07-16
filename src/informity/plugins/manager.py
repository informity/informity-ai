"""Module for plugins manager."""

from __future__ import annotations

# pylint: disable=line-too-long
import json
import re
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from informity.config import APP_DISPLAY_NAME, settings
from informity.utils.directory_utils import ensure_file_directory, ensure_private_file
from informity.utils.json_utils import serialize_config
from informity.version import APP_VERSION

PLUGIN_DIRECTORY_NAME = "plugins"
PLUGIN_MANIFEST_FILENAME = "plugin.json"
PLUGIN_STATE_FILENAME = "plugins-state.json"

PLUGIN_TYPE_VALUES = ("runtime", "specialization")
PLUGIN_HOOK_POINT_VALUES = (
    "prompt_suffix",
    "system_prompt_fragment",
    "query_classification_hint",
    "retrieval_hint",
    "ranking_hint",
    "output_formatting",
    "metadata_extraction",
    "ui_metadata",
)
PLUGIN_PERMISSION_VALUES = (
    "prompt:append",
    "retrieval:hint",
    "retrieval:rank",
    "output:format",
    "metadata:extract",
    "ui:label",
)
PLUGIN_HOOK_PERMISSION_MAP: dict[str, tuple[str, ...]] = {
    "prompt_suffix": ("prompt:append",),
    "system_prompt_fragment": ("prompt:append",),
    "query_classification_hint": ("retrieval:hint",),
    "retrieval_hint": ("retrieval:hint",),
    "ranking_hint": ("retrieval:rank",),
    "output_formatting": ("output:format",),
    "metadata_extraction": ("metadata:extract",),
    "ui_metadata": ("ui:label",),
}
PLUGIN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{1,63}$")
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _normalize_unique_strings(values: list[str]) -> list[str]:
    """Internal helper for normalize unique strings."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


class PluginManifest(BaseModel):
    """Class docstring."""
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    name: str
    version: str
    plugin_type: Literal["runtime", "specialization"]
    entrypoint: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    hook_points: list[str] = Field(default_factory=list)
    min_app_version: str | None = None

    @model_validator(mode="after")
    def _validate_manifest(self) -> PluginManifest:
        """Internal helper for validate manifest."""
        plugin_id = self.id.strip()
        if not plugin_id or not PLUGIN_ID_PATTERN.fullmatch(plugin_id):
            raise ValueError(
                "Plugin id must start with a lowercase letter and use only lowercase letters, "
                "digits, dots, underscores, or hyphens."
            )
        self.id = plugin_id

        name = self.name.strip()
        if not name:
            raise ValueError("Plugin name cannot be empty")
        self.name = name

        version = self.version.strip()
        if not version or not SEMVER_PATTERN.fullmatch(version):
            raise ValueError("Plugin version must use semantic versioning (for example, 1.0.0)")
        self.version = version

        if self.plugin_type not in PLUGIN_TYPE_VALUES:
            raise ValueError(f"Unsupported plugin_type: {self.plugin_type}")

        if self.entrypoint is not None:
            entrypoint = self.entrypoint.strip()
            if not entrypoint:
                self.entrypoint = None
            else:
                entrypoint_path = Path(entrypoint)
                if entrypoint_path.is_absolute() or ".." in entrypoint_path.parts:
                    raise ValueError(
                        "Plugin entrypoint must be a relative path inside the plugin directory"
                    )
                self.entrypoint = entrypoint

        self.capabilities = _normalize_unique_strings(self.capabilities)
        self.permissions = _normalize_unique_strings(self.permissions)
        self.hook_points = _normalize_unique_strings(self.hook_points)

        for hook_point in self.hook_points:
            if hook_point not in PLUGIN_HOOK_POINT_VALUES:
                raise ValueError(f"Unsupported hook point: {hook_point}")
            required_permissions = PLUGIN_HOOK_PERMISSION_MAP.get(hook_point, ())
            missing_permissions = [
                permission
                for permission in required_permissions
                if permission not in self.permissions
            ]
            if missing_permissions:
                raise ValueError(
                    f'Hook point "{hook_point}" requires permissions: {", ".join(missing_permissions)}'
                )

        for permission in self.permissions:
            if permission not in PLUGIN_PERMISSION_VALUES:
                raise ValueError(f"Unsupported permission: {permission}")

        if self.min_app_version is not None:
            min_version = self.min_app_version.strip()
            if not SEMVER_PATTERN.fullmatch(min_version):
                raise ValueError("min_app_version must use semantic versioning")
            self.min_app_version = min_version

        return self


class PluginState(BaseModel):
    """Class docstring."""
    model_config = ConfigDict(extra="forbid")

    global_enabled: bool = True
    enabled_plugin_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> PluginState:
        """Internal helper for normalize."""
        self.enabled_plugin_ids = _normalize_unique_strings(self.enabled_plugin_ids)
        return self


class PluginInventoryItem(BaseModel):
    """Class docstring."""
    id: str
    name: str | None = None
    version: str | None = None
    plugin_type: Literal["runtime", "specialization"] | None = None
    path: str
    installed: bool
    enabled: bool
    valid: bool
    compatible: bool
    loadable: bool
    capabilities: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    hook_points: list[str] = Field(default_factory=list)
    entrypoint: str | None = None
    errors: list[str] = Field(default_factory=list)


class PluginInventoryResponse(BaseModel):
    """Class docstring."""
    app_name: str
    app_version: str
    plugin_root: str
    global_enabled: bool
    total_plugins: int
    enabled_plugins: int
    invalid_plugins: int
    plugins: list[PluginInventoryItem]


class PluginInstallRequest(BaseModel):
    """Class docstring."""
    model_config = ConfigDict(extra="forbid")

    source_path: str
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate_source_path(self) -> PluginInstallRequest:
        """Internal helper for validate source path."""
        path = self.source_path.strip()
        if not path:
            raise ValueError("source_path cannot be empty")
        self.source_path = path
        return self


class PluginStateUpdateRequest(BaseModel):
    """Class docstring."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class PluginGlobalStateUpdateRequest(BaseModel):
    """Class docstring."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool


def get_plugin_root_dir(app_data_dir: Path | None = None) -> Path:
    """Get plugin root dir."""
    base_dir = Path(app_data_dir) if app_data_dir is not None else settings.app_data_dir
    return base_dir / PLUGIN_DIRECTORY_NAME


def get_plugin_state_path(app_data_dir: Path | None = None) -> Path:
    """Get plugin state path."""
    return get_plugin_root_dir(app_data_dir) / PLUGIN_STATE_FILENAME


def _read_json_file(path: Path) -> dict[str, object]:
    """Internal helper for read json file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def load_plugin_state(app_data_dir: Path | None = None) -> PluginState:
    """Load plugin state."""
    state_path = get_plugin_state_path(app_data_dir)
    if not state_path.exists():
        return PluginState()
    payload = _read_json_file(state_path)
    try:
        return PluginState.model_validate(payload)
    except Exception:
        return PluginState()


def save_plugin_state(state: PluginState, app_data_dir: Path | None = None) -> None:
    """Save plugin state."""
    state_path = get_plugin_state_path(app_data_dir)
    ensure_file_directory(state_path)
    temp_path = state_path.with_name(f"{state_path.name}.tmp")
    temp_path.write_text(serialize_config(state.model_dump()), encoding="utf-8")
    ensure_private_file(temp_path)
    temp_path.replace(state_path)
    ensure_private_file(state_path)


def _read_manifest(
    plugin_dir: Path,
    *,
    require_directory_name_match: bool = True,
) -> tuple[PluginManifest | None, list[str]]:
    """Internal helper for read manifest."""
    manifest_path = plugin_dir / PLUGIN_MANIFEST_FILENAME
    if not manifest_path.exists():
        return None, [f"Missing {PLUGIN_MANIFEST_FILENAME}"]
    try:
        payload = _read_json_file(manifest_path)
        manifest = PluginManifest.model_validate(payload)
    except Exception as exc:
        return None, [str(exc)]
    if require_directory_name_match and manifest.id != plugin_dir.name:
        return None, [f'Plugin id "{manifest.id}" must match directory name "{plugin_dir.name}"']
    return manifest, []


def _compare_version_strings(left: str, right: str) -> int:
    """Internal helper for compare version strings."""

    def _parts(version: str) -> tuple[int, int, int]:
        """Internal helper for parts."""
        core = version.split("-", 1)[0].split("+", 1)[0]
        major, minor, patch = (int(part) for part in core.split("."))
        return major, minor, patch

    return (_parts(left) > _parts(right)) - (_parts(left) < _parts(right))


def _is_compatible(manifest: PluginManifest) -> bool:
    """Internal helper for is compatible."""
    if manifest.min_app_version is None:
        return True
    return _compare_version_strings(APP_VERSION, manifest.min_app_version) >= 0


def _build_inventory_item(
    *,
    plugin_dir: Path,
    manifest: PluginManifest | None,
    errors: list[str],
    enabled_plugin_ids: set[str],
    global_enabled: bool,
) -> PluginInventoryItem:
    """Internal helper for build inventory item."""
    plugin_id = manifest.id if manifest is not None else plugin_dir.name
    compatible = bool(manifest is not None and _is_compatible(manifest))
    valid = manifest is not None and not errors
    enabled = bool(
        manifest is not None
        and plugin_id in enabled_plugin_ids
        and global_enabled
        and valid
        and compatible
    )
    return PluginInventoryItem(
        id=plugin_id,
        name=manifest.name if manifest is not None else None,
        version=manifest.version if manifest is not None else None,
        plugin_type=manifest.plugin_type if manifest is not None else None,
        path=str(plugin_dir),
        installed=True,
        enabled=enabled,
        valid=valid,
        compatible=compatible,
        loadable=enabled,
        capabilities=list(manifest.capabilities) if manifest is not None else [],
        permissions=list(manifest.permissions) if manifest is not None else [],
        hook_points=list(manifest.hook_points) if manifest is not None else [],
        entrypoint=manifest.entrypoint if manifest is not None else None,
        errors=errors,
    )


def discover_plugins(app_data_dir: Path | None = None) -> PluginInventoryResponse:
    """Discover plugins."""
    plugin_root = get_plugin_root_dir(app_data_dir)
    state = load_plugin_state(app_data_dir)
    enabled_plugin_ids = set(state.enabled_plugin_ids)
    plugins: list[PluginInventoryItem] = []

    if plugin_root.exists():
        for entry in sorted(plugin_root.iterdir(), key=lambda item: item.name.lower()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            manifest, errors = _read_manifest(entry)
            if manifest is None:
                plugins.append(
                    PluginInventoryItem(
                        id=entry.name,
                        path=str(entry),
                        installed=True,
                        enabled=False,
                        valid=False,
                        compatible=False,
                        loadable=False,
                        errors=errors,
                    )
                )
                continue
            compatible = _is_compatible(manifest)
            if not compatible:
                errors = [f"Plugin requires app version >= {manifest.min_app_version}"]
            plugins.append(
                _build_inventory_item(
                    plugin_dir=entry,
                    manifest=manifest,
                    errors=errors,
                    enabled_plugin_ids=enabled_plugin_ids,
                    global_enabled=state.global_enabled,
                )
            )

    invalid_count = sum(1 for item in plugins if not item.valid)
    enabled_count = sum(1 for item in plugins if item.enabled)

    return PluginInventoryResponse(
        app_name=APP_DISPLAY_NAME,
        app_version=APP_VERSION,
        plugin_root=str(plugin_root),
        global_enabled=state.global_enabled,
        total_plugins=len(plugins),
        enabled_plugins=enabled_count,
        invalid_plugins=invalid_count,
        plugins=plugins,
    )


def install_plugin_from_path(
    source_path: str | Path,
    app_data_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> PluginInventoryResponse:
    """Install plugin from path."""
    source_dir = Path(source_path).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Plugin source directory not found: {source_dir}")

    manifest, errors = _read_manifest(source_dir, require_directory_name_match=False)
    if manifest is None:
        raise ValueError("; ".join(errors) if errors else "Invalid plugin manifest")

    plugin_root = get_plugin_root_dir(app_data_dir)
    plugin_root.mkdir(parents=True, exist_ok=True)
    destination = plugin_root / manifest.id
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Plugin already installed: {manifest.id}")
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)
    return discover_plugins(app_data_dir)


def set_plugin_enabled(
    plugin_id: str,
    enabled: bool,
    app_data_dir: Path | None = None,
) -> PluginInventoryResponse:
    """Set plugin enabled."""
    plugin_id = str(plugin_id or "").strip()
    if not plugin_id:
        raise ValueError("plugin_id cannot be empty")

    inventory = discover_plugins(app_data_dir)
    plugin = next((item for item in inventory.plugins if item.id == plugin_id), None)
    if plugin is None:
        raise KeyError(f"Unknown plugin id: {plugin_id}")
    if not plugin.valid:
        raise ValueError(f"Plugin is not valid: {plugin_id}")

    state = load_plugin_state(app_data_dir)
    enabled_ids = [item for item in state.enabled_plugin_ids if item != plugin_id]
    if enabled:
        enabled_ids.append(plugin_id)
    state.enabled_plugin_ids = enabled_ids
    save_plugin_state(state, app_data_dir)
    return discover_plugins(app_data_dir)


def set_global_plugins_enabled(
    enabled: bool,
    app_data_dir: Path | None = None,
) -> PluginInventoryResponse:
    """Set global plugins enabled."""
    state = load_plugin_state(app_data_dir)
    state.global_enabled = bool(enabled)
    save_plugin_state(state, app_data_dir)
    return discover_plugins(app_data_dir)


def remove_plugin(plugin_id: str, app_data_dir: Path | None = None) -> PluginInventoryResponse:
    """Remove plugin."""
    plugin_id = str(plugin_id or "").strip()
    if not plugin_id:
        raise ValueError("plugin_id cannot be empty")

    plugin_root = get_plugin_root_dir(app_data_dir)
    destination = plugin_root / plugin_id
    if not destination.exists():
        raise KeyError(f"Unknown plugin id: {plugin_id}")
    shutil.rmtree(destination)

    state = load_plugin_state(app_data_dir)
    state.enabled_plugin_ids = [item for item in state.enabled_plugin_ids if item != plugin_id]
    save_plugin_state(state, app_data_dir)
    return discover_plugins(app_data_dir)
