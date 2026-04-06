#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use std::env;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::image::Image as TauriImage;
use tauri::menu::{MenuBuilder, MenuItemBuilder, SubmenuBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};
use tokio::time::sleep;
use uuid::Uuid;

const LOOPBACK_HOST: &str = "127.0.0.1";
const HEALTH_PATH: &str = "/api/health";
const SHUTDOWN_PATH: &str = "/api/shutdown";
const SESSION_HEADER_NAME: &str = "X-Informity-Session";
const BACKEND_STARTUP_STATUS_EVENT: &str = "informity://backend-startup-status";
const BACKEND_RESOURCE_DIR: &str = "backend";
const BACKEND_BINARY_STEM: &str = "informity-backend";
const MANAGED_BACKEND_PID_FILENAME: &str = "informity-ai-backend.pid";
const MANAGED_BACKEND_PID_FILE_ENV: &str = "INFORMITY_MANAGED_PID_FILE";
const TOOLS_DIRNAME: &str = "tools";
const DIAGNOSTICS_DIRNAME: &str = "diagnostics";
const MODELS_DIRNAME: &str = "models";
const CACHE_DIRNAME: &str = "cache";
const CONFIG_FILENAME: &str = "config.json";
const APP_DATA_DIRNAME: &str = ".informity";
const MENU_BAR_TRAY_ID: &str = "informity_menu_bar";
const MENU_BAR_OPEN_MENU_ID: &str = "menu_bar_open";
const MENU_BAR_QUIT_MENU_ID: &str = "menu_bar_quit";
const MENU_BAR_ICON_RELATIVE_PATH: &str = "../icons/trayTemplate.png";
const MENU_BAR_ICON_BYTES: &[u8] = include_bytes!("../icons/trayTemplate.png");
const MENU_ACTION_EVENT: &str = "informity://menu-action";
const MENU_APP_PREFERENCES_ID: &str = "menu_app_preferences";
const MENU_FILE_NEW_CHAT_ID: &str = "menu_file_new_chat";
const MENU_FILE_SCAN_NOW_ID: &str = "menu_file_scan_now";
const MENU_FILE_CLOSE_WINDOW_ID: &str = "menu_file_close_window";
const MENU_VIEW_TOGGLE_SIDEBAR_ID: &str = "menu_view_toggle_sidebar";
const MENU_VIEW_SEARCH_ID: &str = "menu_view_search";
const MENU_VIEW_RELOAD_ID: &str = "menu_view_reload";

#[derive(Default)]
struct BackendController {
    inner: Mutex<BackendRuntime>,
}

#[derive(Default)]
struct BackendRuntime {
    child: Option<Child>,
    base_url: Option<String>,
    session_token: Option<String>,
    pid_file_path: Option<PathBuf>,
    startup_error: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct BackendStartPayload {
    base_url: String,
    session_token: String,
    port: u16,
    launch_mode: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendStatusPayload {
    running: bool,
    base_url: Option<String>,
    startup_error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendStopPayload {
    stopped: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct BackendStartupStatusPayload {
    message: String,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct MenuActionPayload {
    action: String,
}

#[derive(Deserialize)]
struct PersistedDesktopPreferences {
    enable_menu_bar_icon: Option<bool>,
}

struct LaunchSpec {
    program: PathBuf,
    args: Vec<String>,
    current_dir: Option<PathBuf>,
    mode: String,
}

#[tauri::command]
async fn backend_start(
    app: AppHandle,
    controller: State<'_, BackendController>,
) -> Result<BackendStartPayload, String> {
    emit_backend_startup_status(&app, "Starting Informity AI...");

    {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;

        if let Some(child) = guard.child.as_mut() {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    guard.child = None;
                    guard.base_url = None;
                    guard.session_token = None;
                }
                Ok(None) => {
                    if let (Some(base_url), Some(session_token)) =
                        (guard.base_url.clone(), guard.session_token.clone())
                    {
                        let port = parse_port(&base_url)?;
                        return Ok(BackendStartPayload {
                            base_url,
                            session_token,
                            port,
                            launch_mode: "already-running".to_string(),
                        });
                    }
                }
                Err(error) => {
                    guard.startup_error =
                        Some(format!("failed to query backend process state: {error}"));
                }
            }
        }
    }

    let port = pick_free_loopback_port()?;
    let base_url = format!("http://{LOOPBACK_HOST}:{port}");
    let session_token = Uuid::new_v4().to_string();

    let app_data_dir = resolve_managed_app_data_dir(&app)?;
    std::fs::create_dir_all(&app_data_dir)
        .map_err(|error| format!("failed to create app data directory: {error}"))?;
    let pid_file_path = managed_backend_pid_file_path(&app_data_dir);

    emit_backend_startup_status(&app, "Initializing application...");

    {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;
        guard.pid_file_path = Some(pid_file_path.clone());
    }

    // Ensure stale managed backend from previous crashed/force-quit session does not survive.
    if let Err(error) = cleanup_stale_managed_backend(&pid_file_path) {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;
        guard.startup_error = Some(format!("stale backend cleanup warning: {error}"));
    }
    if let Err(error) = cleanup_orphaned_managed_backends(&app_data_dir, &pid_file_path) {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;
        guard.startup_error = Some(format!("orphan backend cleanup warning: {error}"));
    }

    let repo_root = resolve_repo_root_from_manifest().filter(|path| path.exists());
    let cache_dir_override = if env::var_os("INFORMITY_CACHE_DIR").is_none() {
        resolve_managed_cache_dir(&app).ok()
    } else {
        None
    };
    let diagnostics_models_dir_override =
        if env::var_os("INFORMITY_DIAGNOSTICS_MODELS_DIR").is_none() {
            if cfg!(debug_assertions) {
                repo_root
                    .as_ref()
                    .map(|path| {
                        path.join(TOOLS_DIRNAME)
                            .join(DIAGNOSTICS_DIRNAME)
                            .join(MODELS_DIRNAME)
                    })
            } else {
                None
            }
        } else {
            None
        };
    let repo_root_override = if env::var_os("INFORMITY_REPO_ROOT").is_none() {
        repo_root.clone()
    } else {
        None
    };

    let launch_specs = build_launch_specs(&app, repo_root.clone())?;
    let mut launch_errors: Vec<String> = Vec::new();
    let mut launched_mode = String::new();
    let mut launched_child: Option<Child> = None;

    for spec in launch_specs {
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .env("INFORMITY_HOST", LOOPBACK_HOST)
            .env("INFORMITY_PORT", port.to_string())
            .env("INFORMITY_DEV_RELOAD", "false")
            .env("INFORMITY_API_DOCS_ENABLED", "false")
            .env("INFORMITY_APP_DATA_DIR", &app_data_dir)
            .env("INFORMITY_TAURI_SESSION_TOKEN", &session_token)
            .env(MANAGED_BACKEND_PID_FILE_ENV, &pid_file_path);

        if let Some(cache_dir) = &cache_dir_override {
            let _ = std::fs::create_dir_all(cache_dir);
            command.env("INFORMITY_CACHE_DIR", cache_dir);
        }
        if let Some(diagnostics_models_dir) = &diagnostics_models_dir_override {
            command.env("INFORMITY_DIAGNOSTICS_MODELS_DIR", diagnostics_models_dir);
        }
        if let Some(repo_root_path) = &repo_root_override {
            command.env("INFORMITY_REPO_ROOT", repo_root_path);
        }

        if let Some(current_dir) = &spec.current_dir {
            command.current_dir(current_dir);
        }

        match command.spawn() {
            Ok(child) => {
                launched_mode = spec.mode;
                launched_child = Some(child);
                break;
            }
            Err(error) => {
                launch_errors.push(format!(
                    "{} [{} {}]: {}",
                    spec.mode,
                    spec.program.display(),
                    spec.args.join(" "),
                    error
                ));
            }
        }
    }

    let child = launched_child.ok_or_else(|| {
        format!(
            "failed to launch backend runtime. attempts: {}",
            launch_errors.join(" | ")
        )
    })?;

    {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;
        guard.child = Some(child);
        guard.base_url = Some(base_url.clone());
        guard.session_token = Some(session_token.clone());
        guard.pid_file_path = Some(pid_file_path.clone());
        guard.startup_error = None;
    }

    let startup_timeout = if launched_mode == "packaged-sidecar" {
        Duration::from_secs(180)
    } else {
        Duration::from_secs(45)
    };
    let startup_timeout_secs = startup_timeout.as_secs();
    let start = Instant::now();
    while start.elapsed() < startup_timeout {
        if check_health(&base_url, &session_token).await {
            emit_backend_startup_status(&app, "Loading interface...");
            return Ok(BackendStartPayload {
                base_url,
                session_token,
                port,
                launch_mode: launched_mode,
            });
        }

        {
            let mut guard = controller
                .inner
                .lock()
                .map_err(|_| "backend state lock poisoned".to_string())?;
            if let Some(child) = guard.child.as_mut() {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        guard.startup_error = Some(format!(
                            "backend exited before health check completed: {}",
                            status
                        ));
                        guard.child = None;
                        guard.base_url = None;
                        guard.session_token = None;
                        if let Some(path) = guard.pid_file_path.take() {
                            let _ = remove_pid_file(&path);
                        }
                        return Err(guard
                            .startup_error
                            .clone()
                            .unwrap_or_else(|| "backend exited during startup".to_string()));
                    }
                    Ok(None) => {}
                    Err(error) => {
                        guard.startup_error = Some(format!(
                            "failed to inspect backend state during startup: {error}"
                        ));
                    }
                }
            }
        }

        sleep(Duration::from_millis(500)).await;
    }

    let _ = backend_stop_internal(&controller).await;
    {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;
        guard.startup_error = Some(format!(
            "backend health check timed out after {} seconds",
            startup_timeout_secs
        ));
    }
    Err(format!(
        "backend failed to become healthy within {} seconds",
        startup_timeout_secs
    ))
}

fn emit_backend_startup_status(app: &AppHandle, message: &str) {
    let payload = BackendStartupStatusPayload {
        message: message.to_string(),
    };
    let _ = app.emit(BACKEND_STARTUP_STATUS_EVENT, payload);
}

#[tauri::command]
async fn backend_status(
    controller: State<'_, BackendController>,
) -> Result<BackendStatusPayload, String> {
    let mut guard = controller
        .inner
        .lock()
        .map_err(|_| "backend state lock poisoned".to_string())?;

    let running = if let Some(child) = guard.child.as_mut() {
        match child.try_wait() {
            Ok(Some(_)) => {
                guard.child = None;
                guard.base_url = None;
                guard.session_token = None;
                false
            }
            Ok(None) => true,
            Err(error) => {
                guard.startup_error =
                    Some(format!("failed to inspect backend process state: {error}"));
                false
            }
        }
    } else {
        false
    };

    Ok(BackendStatusPayload {
        running,
        base_url: guard.base_url.clone(),
        startup_error: guard.startup_error.clone(),
    })
}

#[tauri::command]
async fn backend_stop(
    controller: State<'_, BackendController>,
) -> Result<BackendStopPayload, String> {
    backend_stop_internal(&controller).await?;
    Ok(BackendStopPayload { stopped: true })
}

async fn backend_stop_internal(controller: &BackendController) -> Result<(), String> {
    let (mut child, base_url, session_token, pid_file_path) = {
        let mut guard = controller
            .inner
            .lock()
            .map_err(|_| "backend state lock poisoned".to_string())?;
        let child = guard.child.take();
        let base_url = guard.base_url.take();
        let session_token = guard.session_token.take();
        let pid_file_path = guard.pid_file_path.take();
        guard.startup_error = None;
        (child, base_url, session_token, pid_file_path)
    };

    let Some(mut child_proc) = child.take() else {
        if let Some(path) = pid_file_path {
            let _ = cleanup_stale_managed_backend(&path);
        }
        return Ok(());
    };

    // Best-effort graceful shutdown first.
    let _ = request_backend_shutdown(base_url.as_deref(), session_token.as_deref()).await;

    // The pid file is authoritative for onefile mode where wrapper PID != backend runtime PID.
    if let Some(path) = pid_file_path.as_ref() {
        if wait_for_managed_backend_exit(path, Duration::from_secs(5)).await {
            let _ = wait_for_child_exit(&mut child_proc, Duration::from_secs(1)).await;
            let _ = remove_pid_file(path);
            return Ok(());
        }

        let _ = terminate_managed_backend_from_pid_file(path);
        let _ = wait_for_managed_backend_exit(path, Duration::from_secs(2)).await;
        let _ = remove_pid_file(path);
    }

    if let Ok(false) = wait_for_child_exit(&mut child_proc, Duration::from_millis(800)).await {
        let _ = child_proc.kill();
        let _ = child_proc.wait();
    }

    if let Some(path) = pid_file_path {
        let _ = cleanup_stale_managed_backend(&path);
    }
    Ok(())
}

fn build_launch_specs(
    app: &AppHandle,
    repo_root: Option<PathBuf>,
) -> Result<Vec<LaunchSpec>, String> {
    let mut specs = Vec::new();

    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|error| format!("failed to resolve resource directory: {error}"))?;

    let sidecar = resolve_packaged_sidecar_program(&resource_dir)?;
    specs.push(LaunchSpec {
        program: sidecar,
        args: Vec::new(),
        current_dir: if cfg!(debug_assertions) {
            repo_root.clone()
        } else {
            None
        },
        mode: "packaged-sidecar".to_string(),
    });

    if cfg!(debug_assertions) {
        if let Some(repo_root) = repo_root {
            let venv_python = repo_root.join(".venv").join("bin").join("python");
            if venv_python.exists() {
                specs.insert(
                    0,
                    LaunchSpec {
                        program: venv_python,
                        args: vec!["-m".to_string(), "informity.main".to_string()],
                        current_dir: Some(repo_root.clone()),
                        mode: "dev-venv-python".to_string(),
                    },
                );
            }

            specs.insert(
                specs.len().saturating_sub(1),
                LaunchSpec {
                    program: PathBuf::from("uv"),
                    args: vec![
                        "run".to_string(),
                        "python".to_string(),
                        "-m".to_string(),
                        "informity.main".to_string(),
                    ],
                    current_dir: Some(repo_root),
                    mode: "dev-uv-run".to_string(),
                },
            );
        }
    }

    Ok(specs)
}

fn resolve_packaged_sidecar_program(resource_dir: &Path) -> Result<PathBuf, String> {
    let sidecar_root = resource_dir.join(BACKEND_RESOURCE_DIR);
    let binary_name = platform_backend_binary_name();
    let bundle_dir = sidecar_root
        .join("informity-backend-bundle")
        .join(&binary_name);

    // Required onedir layout with stable bundle directory:
    // resources/backend/informity-backend-bundle/<binary-name>
    if bundle_dir.exists() {
        return Ok(bundle_dir);
    }

    Err(format!(
        "backend sidecar not found (checked: {})",
        bundle_dir.display(),
    ))
}

fn resolve_repo_root_from_manifest() -> Option<PathBuf> {
    let mut current = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if current.join("pyproject.toml").exists() {
            return Some(current);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

fn resolve_managed_app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(home) = env::var_os("HOME").map(PathBuf::from) {
        return Ok(home.join(APP_DATA_DIRNAME));
    }
    if let Some(home) = env::var_os("USERPROFILE").map(PathBuf::from) {
        return Ok(home.join(APP_DATA_DIRNAME));
    }

    app.path()
        .app_data_dir()
        .map_err(|error| format!("failed to resolve app data directory: {error}"))
}

fn resolve_managed_cache_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let app_data_dir = resolve_managed_app_data_dir(app)?;
    Ok(app_data_dir.join(CACHE_DIRNAME))
}

fn managed_backend_pid_file_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join(MANAGED_BACKEND_PID_FILENAME)
}

fn remove_pid_file(path: &Path) -> Result<(), String> {
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|error| format!("failed to remove pid file {}: {error}", path.display()))?;
    }
    Ok(())
}

fn read_managed_backend_pid(path: &Path) -> Result<Option<u32>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read pid file {}: {error}", path.display()))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let pid = trimmed
        .parse::<u32>()
        .map_err(|error| format!("invalid pid file contents in {}: {error}", path.display()))?;
    Ok(Some(pid))
}

fn cleanup_stale_managed_backend(path: &Path) -> Result<(), String> {
    let pid = match read_managed_backend_pid(path)? {
        Some(value) => value,
        None => return Ok(()),
    };

    if !is_process_alive(pid) {
        remove_pid_file(path)?;
        return Ok(());
    }

    if !is_managed_backend_process(pid)? {
        return Err(format!(
            "managed backend pid file points to unexpected live process (pid={pid})"
        ));
    }

    terminate_process(pid)?;
    remove_pid_file(path)?;
    Ok(())
}

fn cleanup_orphaned_managed_backends(
    app_data_dir: &Path,
    managed_pid_file: &Path,
) -> Result<(), String> {
    #[cfg(unix)]
    {
        let protected_pid = read_managed_backend_pid(managed_pid_file).ok().flatten();
        let marker = format!("INFORMITY_APP_DATA_DIR={}", app_data_dir.display());
        let process_list = Command::new("ps")
            .args(["-axo", "pid=,command="])
            .output()
            .map_err(|error| format!("failed to enumerate processes: {error}"))?;
        if !process_list.status.success() {
            return Ok(());
        }

        for line in String::from_utf8_lossy(&process_list.stdout).lines() {
            let trimmed = line.trim_start();
            if trimmed.is_empty() {
                continue;
            }

            let mut parts = trimmed.splitn(2, char::is_whitespace);
            let Some(pid_str) = parts.next() else {
                continue;
            };
            let command = parts.next().unwrap_or_default().trim();
            let Ok(pid) = pid_str.parse::<u32>() else {
                continue;
            };
            if Some(pid) == protected_pid {
                continue;
            }
            if !command.contains("informity-backend") {
                continue;
            }

            let env_output = Command::new("ps")
                .args(["eww", "-p", &pid.to_string()])
                .output()
                .map_err(|error| format!("failed to inspect process environment for pid {pid}: {error}"))?;
            if !env_output.status.success() {
                continue;
            }
            let env_text = String::from_utf8_lossy(&env_output.stdout);
            if !env_text.contains(&marker) {
                continue;
            }
            if is_managed_backend_process(pid)? {
                terminate_process(pid)?;
            }
        }
    }

    Ok(())
}

async fn request_backend_shutdown(
    base_url: Option<&str>,
    session_token: Option<&str>,
) -> Result<(), String> {
    let Some(url) = base_url else {
        return Ok(());
    };

    let shutdown_url = format!("{}{}", url, SHUTDOWN_PATH);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| format!("failed to build HTTP client: {error}"))?;

    let mut request = client.post(shutdown_url);
    if let Some(token) = session_token {
        request = request.header(SESSION_HEADER_NAME, token);
    }

    let _ = request.send().await;
    Ok(())
}

async fn wait_for_child_exit(child: &mut Child, timeout: Duration) -> Result<bool, String> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        match child.try_wait() {
            Ok(Some(_status)) => return Ok(true),
            Ok(None) => sleep(Duration::from_millis(150)).await,
            Err(error) => return Err(format!("failed to poll backend wrapper process: {error}")),
        }
    }
    Ok(false)
}

async fn wait_for_managed_backend_exit(path: &Path, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        match read_managed_backend_pid(path) {
            Ok(None) => return true,
            Ok(Some(pid)) => {
                if !is_process_alive(pid) {
                    let _ = remove_pid_file(path);
                    return true;
                }
            }
            Err(_) => {}
        }
        sleep(Duration::from_millis(150)).await;
    }
    false
}

fn terminate_managed_backend_from_pid_file(path: &Path) -> Result<(), String> {
    let Some(pid) = read_managed_backend_pid(path)? else {
        return Ok(());
    };

    if !is_process_alive(pid) {
        remove_pid_file(path)?;
        return Ok(());
    }

    if !is_managed_backend_process(pid)? {
        return Err(format!(
            "refusing to terminate unexpected process from pid file (pid={pid})"
        ));
    }

    terminate_process(pid)?;
    remove_pid_file(path)?;
    Ok(())
}

fn is_managed_backend_process(pid: u32) -> Result<bool, String> {
    #[cfg(unix)]
    {
        let output = Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "command="])
            .output()
            .map_err(|error| format!("failed to inspect process command for pid {pid}: {error}"))?;
        if !output.status.success() {
            return Ok(false);
        }
        let command = String::from_utf8_lossy(&output.stdout);
        return Ok(command.contains("informity.main")
            || command.contains("informity-backend")
            || command.contains("informity-ai")
            || command.contains("Informity AI"));
    }

    #[cfg(windows)]
    {
        let output = Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .map_err(|error| format!("failed to inspect task list for pid {pid}: {error}"))?;
        if !output.status.success() {
            return Ok(false);
        }
        let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
        Ok(stdout.contains("python") || stdout.contains("informity-backend"))
    }
}

fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[cfg(windows)]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .map(|output| {
                if !output.status.success() {
                    return false;
                }
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.lines().any(|line| line.contains(&pid.to_string()))
            })
            .unwrap_or(false)
    }
}

fn terminate_process(pid: u32) -> Result<(), String> {
    #[cfg(unix)]
    {
        let term_status = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .map_err(|error| format!("failed to send SIGTERM to pid {pid}: {error}"))?;
        if !term_status.success() {
            return Err(format!("SIGTERM command failed for pid {pid}"));
        }

        let wait_deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < wait_deadline {
            if !is_process_alive(pid) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(150));
        }

        let kill_status = Command::new("kill")
            .args(["-KILL", &pid.to_string()])
            .status()
            .map_err(|error| format!("failed to send SIGKILL to pid {pid}: {error}"))?;
        if !kill_status.success() {
            return Err(format!("SIGKILL command failed for pid {pid}"));
        }
        Ok(())
    }

    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
            .map_err(|error| format!("failed to terminate pid {pid}: {error}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("taskkill failed for pid {pid}"))
        }
    }
}

fn platform_backend_binary_name() -> String {
    if cfg!(target_os = "windows") {
        format!("{}.exe", BACKEND_BINARY_STEM)
    } else {
        BACKEND_BINARY_STEM.to_string()
    }
}

fn pick_free_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind((LOOPBACK_HOST, 0))
        .map_err(|error| format!("failed to allocate free loopback port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| format!("failed to inspect allocated loopback port: {error}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn parse_port(base_url: &str) -> Result<u16, String> {
    let (_, port_str) = base_url
        .rsplit_once(':')
        .ok_or_else(|| format!("invalid backend base URL (missing port): {base_url}"))?;
    port_str
        .parse::<u16>()
        .map_err(|error| format!("invalid backend port in URL {base_url}: {error}"))
}

async fn check_health(base_url: &str, session_token: &str) -> bool {
    let health_url = format!("{}{}", base_url, HEALTH_PATH);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client
        .get(health_url)
        .header(SESSION_HEADER_NAME, session_token)
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn emit_menu_action(app: &AppHandle, action: &str) {
    let payload = MenuActionPayload {
        action: action.to_string(),
    };
    let _ = app.emit(MENU_ACTION_EVENT, payload);
}

fn menu_bar_icon_enabled(app: &AppHandle) -> bool {
    let app_data_dir = match resolve_managed_app_data_dir(app) {
        Ok(path) => path,
        Err(_) => return false,
    };
    let config_path = app_data_dir.join(CONFIG_FILENAME);
    let raw = match std::fs::read_to_string(config_path) {
        Ok(contents) => contents,
        Err(_) => return false,
    };
    match serde_json::from_str::<PersistedDesktopPreferences>(&raw) {
        Ok(settings) => settings.enable_menu_bar_icon.unwrap_or(false),
        Err(_) => false,
    }
}

fn ensure_menu_bar_icon(app: &AppHandle) -> Result<(), String> {
    if app.tray_by_id(MENU_BAR_TRAY_ID).is_some() {
        return Ok(());
    }

    let tray_icon = app
        .path()
        .resource_dir()
        .ok()
        .map(|path| path.join("icons").join("trayTemplate.png"))
        .and_then(|path| TauriImage::from_path(path).ok())
        .or_else(|| TauriImage::from_bytes(MENU_BAR_ICON_BYTES).ok())
        .or_else(|| app.default_window_icon().cloned())
        .or_else(|| {
            let tray_icon_path =
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MENU_BAR_ICON_RELATIVE_PATH);
            TauriImage::from_path(tray_icon_path).ok()
        });
    let tray_icon = tray_icon
        .ok_or_else(|| "failed to initialize menu bar icon: no suitable icon image found".to_string())?;

    let open_item = match MenuItemBuilder::with_id(MENU_BAR_OPEN_MENU_ID, "Open Informity AI")
        .build(app)
    {
        Ok(item) => item,
        Err(error) => {
            return Err(format!("failed to initialize menu bar icon open item: {error}"));
        }
    };
    let quit_item = match MenuItemBuilder::with_id(MENU_BAR_QUIT_MENU_ID, "Quit Informity AI")
        .build(app)
    {
        Ok(item) => item,
        Err(error) => {
            return Err(format!("failed to initialize menu bar icon quit item: {error}"));
        }
    };
    let tray_menu = match MenuBuilder::new(app)
        .item(&open_item)
        .separator()
        .item(&quit_item)
        .build()
    {
        Ok(menu) => menu,
        Err(error) => {
            return Err(format!("failed to initialize menu bar icon menu: {error}"));
        }
    };

    let tray_icon = TrayIconBuilder::with_id(MENU_BAR_TRAY_ID)
        .icon(tray_icon)
        .icon_as_template(true)
        .tooltip("Informity AI")
        .show_menu_on_left_click(false)
        .menu(&tray_menu)
        .on_menu_event(
            |app, event: tauri::menu::MenuEvent| match event.id().as_ref() {
                MENU_BAR_OPEN_MENU_ID => show_main_window(app),
                MENU_BAR_QUIT_MENU_ID => app.exit(0),
                _ => {}
            },
        )
        .on_tray_icon_event(|tray: &tauri::tray::TrayIcon<_>, event: TrayIconEvent| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        })
        .build(app)
        .map_err(|error| format!("failed to initialize menu bar icon: {error}"))?;

    tray_icon
        .set_visible(true)
        .map_err(|error| format!("failed to show menu bar icon: {error}"))?;

    Ok(())
}

fn set_menu_bar_icon_visibility(app: &AppHandle, enabled: bool) -> Result<(), String> {
    if enabled {
        ensure_menu_bar_icon(app)?;
        return Ok(());
    }

    let _ = app.remove_tray_by_id(MENU_BAR_TRAY_ID);
    Ok(())
}

fn setup_menu_bar_icon(app: &AppHandle) {
    let enabled = menu_bar_icon_enabled(app);
    if let Err(error) = set_menu_bar_icon_visibility(app, enabled) {
        eprintln!("[menu-bar-icon] initial setup failed: {error}");
        if enabled {
            let app_handle = app.clone();
            tauri::async_runtime::spawn(async move {
                sleep(Duration::from_millis(900)).await;
                if let Err(retry_error) = set_menu_bar_icon_visibility(&app_handle, true) {
                    eprintln!("[menu-bar-icon] retry setup failed: {retry_error}");
                }
            });
        }
    } else {
        eprintln!("[menu-bar-icon] initial setup applied (enabled={enabled})");
    }
}

#[tauri::command]
fn set_menu_bar_icon_enabled(app: AppHandle, enabled: bool) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let result = set_menu_bar_icon_visibility(&app, enabled);
        if let Err(error) = &result {
            eprintln!("[menu-bar-icon] toggle failed (enabled={enabled}): {error}");
        } else {
            eprintln!("[menu-bar-icon] toggle applied (enabled={enabled})");
        }
        return result;
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        let _ = enabled;
        Ok(())
    }
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .enable_macos_default_menu(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            MENU_APP_PREFERENCES_ID => emit_menu_action(app, "preferences"),
            MENU_FILE_NEW_CHAT_ID => emit_menu_action(app, "new-chat"),
            MENU_FILE_SCAN_NOW_ID => emit_menu_action(app, "scan-now"),
            MENU_FILE_CLOSE_WINDOW_ID => app.exit(0),
            MENU_VIEW_TOGGLE_SIDEBAR_ID => emit_menu_action(app, "toggle-sidebar"),
            MENU_VIEW_SEARCH_ID => emit_menu_action(app, "focus-search"),
            MENU_VIEW_RELOAD_ID => {
                #[cfg(debug_assertions)]
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.eval("window.location.reload()");
                }
            }
            _ => {}
        })
        .setup(|app| {
            #[cfg(target_os = "macos")]
            {
                let preferences_item =
                    MenuItemBuilder::with_id(MENU_APP_PREFERENCES_ID, "Preferences…")
                        .accelerator("Cmd+,")
                        .build(app)?;
                let app_submenu = SubmenuBuilder::new(app, "Informity AI")
                    .about_with_text("About Informity AI", None)
                    .separator()
                    .item(&preferences_item)
                    .separator()
                    .services()
                    .separator()
                    .hide_with_text("Hide Informity AI")
                    .hide_others()
                    .show_all()
                    .separator()
                    .quit_with_text("Quit Informity AI")
                    .build()?;

                let new_chat_item =
                    MenuItemBuilder::with_id(MENU_FILE_NEW_CHAT_ID, "New Chat")
                        .accelerator("Cmd+N")
                        .build(app)?;
                let scan_now_item =
                    MenuItemBuilder::with_id(MENU_FILE_SCAN_NOW_ID, "Scan Now").build(app)?;
                let close_window_item =
                    MenuItemBuilder::with_id(MENU_FILE_CLOSE_WINDOW_ID, "Close Window")
                        .accelerator("Cmd+W")
                        .build(app)?;
                let file_submenu = SubmenuBuilder::new(app, "File")
                    .item(&new_chat_item)
                    .item(&scan_now_item)
                    .separator()
                    .item(&close_window_item)
                    .build()?;

                let edit_submenu = SubmenuBuilder::new(app, "Edit")
                    .undo()
                    .redo()
                    .separator()
                    .cut()
                    .copy()
                    .paste()
                    .select_all()
                    .build()?;

                let toggle_sidebar_item =
                    MenuItemBuilder::with_id(MENU_VIEW_TOGGLE_SIDEBAR_ID, "Toggle Sidebar")
                        .accelerator("Cmd+B")
                        .build(app)?;
                let search_item = MenuItemBuilder::with_id(MENU_VIEW_SEARCH_ID, "Search")
                    .accelerator("Cmd+K")
                    .build(app)?;
                let view_submenu = {
                    let builder = SubmenuBuilder::new(app, "View")
                        .item(&toggle_sidebar_item)
                        .item(&search_item)
                        .separator();
                    #[cfg(debug_assertions)]
                    let builder = {
                        let reload_item = MenuItemBuilder::with_id(MENU_VIEW_RELOAD_ID, "Reload")
                            .accelerator("Cmd+R")
                            .build(app)?;
                        builder.item(&reload_item)
                    };
                    builder.fullscreen().build()?
                };

                let window_submenu = SubmenuBuilder::new(app, "Window")
                    .minimize()
                    .maximize()
                    .separator()
                    .close_window()
                    .build()?;

                let docs_item = MenuItemBuilder::new("Documentation (Coming Soon)").build(app)?;
                docs_item.set_enabled(false)?;
                let report_item = MenuItemBuilder::new("Report Issue (Coming Soon)").build(app)?;
                report_item.set_enabled(false)?;
                let updates_item =
                    MenuItemBuilder::new("Check for Updates (Coming Soon)").build(app)?;
                updates_item.set_enabled(false)?;
                let help_submenu = SubmenuBuilder::new(app, "Help")
                    .item(&docs_item)
                    .item(&report_item)
                    .item(&updates_item)
                    .build()?;

                let menu = MenuBuilder::new(app)
                    .item(&app_submenu)
                    .item(&file_submenu)
                    .item(&edit_submenu)
                    .item(&view_submenu)
                    .item(&window_submenu)
                    .item(&help_submenu)
                    .build()?;

                app.set_menu(menu)?;
            }

            Ok(())
        })
        .manage(BackendController::default())
        .invoke_handler(tauri::generate_handler![
            backend_start,
            backend_status,
            backend_stop,
            set_menu_bar_icon_enabled
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| {
        match event {
            #[cfg(target_os = "macos")]
            RunEvent::Ready => {
                setup_menu_bar_icon(app);
            }
            RunEvent::ExitRequested { .. } | RunEvent::Exit => {
                let controller = app.state::<BackendController>();
                let _ = tauri::async_runtime::block_on(backend_stop_internal(&controller));
            }
            _ => {}
        }
    });
}
