"""Module for api routes plugins."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from informity.plugins.manager import (
    PluginGlobalStateUpdateRequest,
    PluginInstallRequest,
    PluginInventoryResponse,
    PluginStateUpdateRequest,
    discover_plugins,
    install_plugin_from_path,
    remove_plugin,
    set_global_plugins_enabled,
    set_plugin_enabled,
)

router = APIRouter(tags=["plugins"])


@router.get("/api/plugins", response_model=PluginInventoryResponse)
async def list_plugins() -> PluginInventoryResponse:
    """List plugins."""
    return discover_plugins()


@router.post("/api/plugins/install", response_model=PluginInventoryResponse)
async def install_plugin(request: PluginInstallRequest) -> PluginInventoryResponse:
    """Install plugin."""
    try:
        return install_plugin_from_path(request.source_path, overwrite=bool(request.overwrite))
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/api/plugins/{plugin_id}", response_model=PluginInventoryResponse)
async def update_plugin_state(
    plugin_id: str, request: PluginStateUpdateRequest
) -> PluginInventoryResponse:
    """Update plugin state."""
    try:
        return set_plugin_enabled(plugin_id, request.enabled)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/plugins/{plugin_id}", response_model=PluginInventoryResponse)
async def delete_plugin(plugin_id: str) -> PluginInventoryResponse:
    """Delete plugin."""
    try:
        return remove_plugin(plugin_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/api/plugins", response_model=PluginInventoryResponse)
async def update_plugin_global_state(
    request: PluginGlobalStateUpdateRequest,
) -> PluginInventoryResponse:
    """Update plugin global state."""
    try:
        return set_global_plugins_enabled(request.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
