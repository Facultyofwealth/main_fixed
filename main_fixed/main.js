'use strict';
/**
 * In The Beginning — Electron desktop shell
 * ------------------------------------------------------------------
 * Replaces the pywebview launcher. Two responsibilities:
 *
 *   1. MULTI-MONITOR OUTPUT. Electron's `screen` module reports exact
 *      display bounds, and a frameless BrowserWindow created AT those
 *      bounds physically occupies that monitor before anything else
 *      happens. This is why the pywebview version failed: its
 *      move()/resize() calls were queued asynchronously while
 *      toggle_fullscreen() fired immediately, so WebView2 fullscreened
 *      on whichever monitor Windows still considered the owner (the
 *      primary) and the pending move was discarded.
 *
 *   2. OS-LEVEL PERMISSIONS. Microphone and window creation are granted
 *      by the app itself via Electron's permission handlers + Windows
 *      privacy settings, instead of a browser consent prompt.
 */

const { app, BrowserWindow, screen, ipcMain, session, systemPreferences, shell, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { spawn } = require('child_process');
const { autoUpdater } = require('electron-updater');

// ─────────────────────────── configuration ───────────────────────────
const BACKEND_PORT = parseInt(process.env.ITB_PORT || '8000', 10);
const BACKEND_HOST = '127.0.0.1';
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;

// Both Electron and Python must agree on where output_settings.json
// lives, so we pin ITB_DATA_DIR and pass it to the Python child.
const DATA_DIR = process.env.ITB_DATA_DIR || app.getPath('userData');
const OUTPUT_SETTINGS_PATH = path.join(DATA_DIR, 'output_settings.json');

let controlWin = null;
let outputWin = null;
let backendProc = null;
let isQuitting = false;

// ─────────────────────────── settings I/O ────────────────────────────
function readOutputSettings() {
  try {
    const raw = fs.readFileSync(OUTPUT_SETTINGS_PATH, 'utf8');
    const data = JSON.parse(raw);
    return (data && typeof data === 'object') ? data : {};
  } catch (_) {
    return {};
  }
}

function writeOutputSettings(data) {
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(OUTPUT_SETTINGS_PATH, JSON.stringify(data, null, 2), 'utf8');
    return true;
  } catch (e) {
    console.error('[itb] could not save output settings:', e.message);
    return false;
  }
}

function getTarget() {
  const v = parseInt(readOutputSettings().screen_index, 10);
  return Number.isFinite(v) ? v : -1;
}

function setTarget(idx) {
  const settings = readOutputSettings();
  settings.screen_index = parseInt(idx, 10);
  writeOutputSettings(settings);
  return settings.screen_index;
}

// ─────────────────────────── display listing ─────────────────────────
/**
 * Ordering MUST mirror the Python list_monitors(): primary first, then
 * left-to-right / top-to-bottom. The saved screen_index refers to a
 * position in this list, so if the two disagree the output lands on the
 * wrong monitor.
 */
function listDisplays() {
  const primaryId = screen.getPrimaryDisplay().id;
  const mons = screen.getAllDisplays().map((d) => ({
    id: d.id,
    x: d.bounds.x,
    y: d.bounds.y,
    width: d.bounds.width,
    height: d.bounds.height,
    scaleFactor: d.scaleFactor,
    is_primary: d.id === primaryId,
  }));

  mons.sort((a, b) => {
    if (a.is_primary !== b.is_primary) return a.is_primary ? -1 : 1;
    return (a.x - b.x) || (a.y - b.y);
  });

  return mons.map((m, i) => ({
    index: i,
    label: (m.is_primary ? 'Primary' : `Screen ${i + 1}`) + ` — ${m.width}x${m.height}`,
    ...m,
  }));
}

function resolveTargetDisplay() {
  const mons = listDisplays();
  if (!mons.length) return null;
  let idx = getTarget();
  if (idx < 0 || idx >= mons.length) {
    // Nothing valid saved — prefer the first NON-primary display (the TV)
    // so the output never covers the operator's controls.
    const ext = mons.find((m) => !m.is_primary);
    idx = ext ? ext.index : 0;
  }
  return { monitor: mons[idx], index: idx, monitors: mons };
}

// ─────────────────────────── output window ───────────────────────────
function createOutputWindow(churchId) {
  if (outputWin && !outputWin.isDestroyed()) {
    return { ok: true, open: true, already_open: true, screen_index: getTarget() };
  }

  const resolved = resolveTargetDisplay();
  if (!resolved) return { ok: false, error: 'No monitors detected.' };

  const t = resolved.monitor;

  outputWin = new BrowserWindow({
    x: t.x,
    y: t.y,
    width: t.width,
    height: t.height,
    frame: false,
    show: false,
    resizable: false,
    movable: false,
    minimizable: false,
    maximizable: false,
    fullscreenable: true,
    skipTaskbar: true,
    autoHideMenuBar: true,
    backgroundColor: '#000000',
    title: 'In The Beginning — Output',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false, // keep rendering while controls have focus
    },
  });

  outputWin.setMenuBarVisibility(false);
  outputWin.loadURL(`${BACKEND_URL}/display?church=${encodeURIComponent(churchId || '')}&chrome=0`);

  outputWin.once('ready-to-show', () => {
    if (!outputWin || outputWin.isDestroyed()) return;
    const bounds = { x: t.x, y: t.y, width: t.width, height: t.height };

    // Re-assert bounds, then show WITHOUT stealing focus from the controls.
    outputWin.setBounds(bounds);
    outputWin.showInactive();

    // Float above the secondary monitor's taskbar.
    outputWin.setAlwaysOnTop(true, 'screen-saver');

    // Only now attempt true fullscreen — and verify it stayed put. On
    // Windows, setFullScreen can relocate a window to the primary display;
    // if that happens we roll back to exact frameless bounds, which is
    // visually identical and cannot jump.
    setTimeout(() => {
      if (!outputWin || outputWin.isDestroyed()) return;
      try {
        outputWin.setFullScreen(true);
        setTimeout(() => {
          if (!outputWin || outputWin.isDestroyed()) return;
          const nowOn = screen.getDisplayMatching(outputWin.getBounds());
          if (nowOn.id !== t.id) {
            console.warn('[itb] fullscreen moved the window off target — reverting to exact bounds');
            outputWin.setFullScreen(false);
            outputWin.setBounds(bounds);
          }
        }, 150);
      } catch (e) {
        try { outputWin.setBounds(bounds); } catch (_) {}
      }
    }, 120);

    console.log(`[itb] output window opened on ${t.label} at (${t.x},${t.y})`);
  });

  outputWin.on('closed', () => {
    // Clears the handle ONLY. The saved screen_index is never reset here,
    // so turning the output back on reuses the same monitor.
    outputWin = null;
  });

  return { ok: true, open: true, screen_index: resolved.index, screen: t };
}

function destroyOutputWindow() {
  if (outputWin && !outputWin.isDestroyed()) {
    outputWin.destroy();
  }
  outputWin = null;
  return { ok: true, open: false, screen_index: getTarget() };
}

// ─────────────────────────── control window ──────────────────────────
function createControlWindow() {
  const primary = screen.getPrimaryDisplay();
  const { x, y, width, height } = primary.workArea;

  controlWin = new BrowserWindow({
    x,
    y,
    width,
    height,
    show: false,
    backgroundColor: '#0f0f1a',
    title: 'In The Beginning',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });

  controlWin.setMenuBarVisibility(false);
  controlWin.loadURL(BACKEND_URL);

  controlWin.once('ready-to-show', () => {
    controlWin.maximize();
    controlWin.show();
  });

  controlWin.on('closed', () => {
    controlWin = null;
    if (!isQuitting) app.quit();
  });

  // Keep stray target=_blank links inside the app / in the real browser.
  controlWin.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith(BACKEND_URL)) return { action: 'allow' };
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// ─────────────────────────── permissions ─────────────────────────────
function configurePermissions() {
  const ses = session.defaultSession;

  // Grant media + fullscreen to our own local backend without a browser
  // consent prompt. The OS-level gate is handled separately below.
  const ALLOWED = new Set([
    'media', 'audioCapture', 'videoCapture', 'display-capture',
    'fullscreen', 'clipboard-read', 'clipboard-sanitized-write', 'notifications',
  ]);

  ses.setPermissionRequestHandler((wc, permission, callback) => {
    const url = wc && wc.getURL ? wc.getURL() : '';
    const local = url.startsWith(BACKEND_URL) || url.startsWith('http://localhost');
    callback(local && ALLOWED.has(permission));
  });

  ses.setPermissionCheckHandler((wc, permission, origin) => {
    const local = (origin || '').startsWith(BACKEND_URL) || (origin || '').startsWith('http://localhost');
    return local && ALLOWED.has(permission);
  });

  // Auto-pick the default microphone so getUserMedia never stalls.
  if (ses.setDisplayMediaRequestHandler) {
    ses.setDisplayMediaRequestHandler((request, callback) => {
      callback({ audio: 'loopback' });
    }, { useSystemPicker: true });
  }
}

/**
 * Windows gates microphone access at the OS level (Settings → Privacy →
 * Microphone). Electron can read that state; macOS can request it outright.
 */
async function ensureMicrophoneAccess() {
  try {
    if (process.platform === 'darwin') {
      const status = systemPreferences.getMediaAccessStatus('microphone');
      if (status !== 'granted') await systemPreferences.askForMediaAccess('microphone');
      return systemPreferences.getMediaAccessStatus('microphone');
    }
    if (process.platform === 'win32') {
      return systemPreferences.getMediaAccessStatus('microphone');
    }
  } catch (_) {}
  return 'unknown';
}

// ─────────────────────────── python backend ──────────────────────────
function backendIsUp() {
  return new Promise((resolve) => {
    const req = http.get(`${BACKEND_URL}/output/status`, { timeout: 1000 }, (res) => {
      res.resume();
      resolve(res.statusCode > 0);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

function resolveBackendCommand() {
  // Frozen PyInstaller exe shipped next to the Electron app
  const exeName = process.platform === 'win32' ? 'main_fixed.exe' : 'main_fixed';
  const candidates = [
    path.join(process.resourcesPath || __dirname, 'backend', exeName),
    path.join(__dirname, 'backend', exeName),
    path.join(__dirname, 'dist', exeName),
  ];
  for (const exe of candidates) {
    if (fs.existsSync(exe)) return { cmd: exe, args: [], cwd: path.dirname(exe) };
  }
  // Development: run the script with the interpreter
  const script = path.join(__dirname, 'main_fixed.py');
  if (fs.existsSync(script)) {
    const py = process.env.ITB_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
    // -u: force stdout/stderr unbuffered. Without it, Python buffers output
    // when its stdout is a pipe (as it is under Electron's spawn) rather than
    // a real terminal, so log lines - including "Application startup
    // complete." - don't appear until the buffer fills or the process exits,
    // which looks exactly like a hang.
    return { cmd: py, args: ['-u', script], cwd: __dirname };
  }
  return null;
}

async function startBackend() {
  if (await backendIsUp()) {
    console.log('[itb] backend already running — attaching');
    return true;
  }

  const spec = resolveBackendCommand();
  if (!spec) {
    dialog.showErrorBox('In The Beginning',
      'Could not locate the Python backend (main_fixed.exe or main_fixed.py).');
    return false;
  }

  console.log(`[itb] starting backend: ${spec.cmd} ${spec.args.join(' ')}`);
  backendProc = spawn(spec.cmd, spec.args, {
    cwd: spec.cwd,
    env: {
      ...process.env,
      ITB_ELECTRON: '1',          // tells Python to stay headless (no pywebview)
      ITB_DATA_DIR: DATA_DIR,     // both sides share output_settings.json
      PORT: String(BACKEND_PORT),
      PYTHONUNBUFFERED: '1',      // belt-and-suspenders alongside the -u flag above
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  backendProc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  backendProc.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  backendProc.on('exit', (code) => {
    console.log(`[itb] backend exited with code ${code}`);
    backendProc = null;
    if (!isQuitting) {
      dialog.showErrorBox('In The Beginning', `The backend stopped unexpectedly (code ${code}).`);
      app.quit();
    }
  });

  // Poll until uvicorn binds the port (up to ~90s — this backend loads a
  // 31k-verse Bible JSON plus FastAPI/uvicorn's own cold-start cost, which
  // can exceed 30s; start_in_the_beginning.py already gives it 90s).
  for (let i = 0; i < 180; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (await backendIsUp()) {
      console.log('[itb] backend is up');
      return true;
    }
  }
  dialog.showErrorBox('In The Beginning', 'The backend did not start in time.');
  return false;
}

function stopBackend() {
  if (backendProc) {
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(backendProc.pid), '/f', '/t']);
      } else {
        backendProc.kill('SIGTERM');
      }
    } catch (_) {}
    backendProc = null;
  }
}

// ─────────────────────────── auto-update ─────────────────────────────
/**
 * Checks the GitHub Releases page named in package.json ("build.publish")
 * for a newer version. The popup itself is drawn INSIDE the app page (see
 * In_the_Beginning.html) so it can use the app's own colours — a native
 * dialog.showMessageBox cannot be styled.
 *
 * Flow:  idle -> available -> downloading -> ready -> (restart & install)
 * Nothing is downloaded or installed until the operator clicks a button.
 *
 * Dev testing without publishing anything:
 *   set ITB_FAKE_UPDATE=1   (PowerShell:  $env:ITB_FAKE_UPDATE="1")
 *   npm start
 */
const FAKE_UPDATE = !app.isPackaged && process.env.ITB_FAKE_UPDATE === '1';
const UPDATE_CHECK_DELAY_MS = 8 * 1000;               // let the UI settle first
const UPDATE_CHECK_EVERY_MS = 4 * 60 * 60 * 1000;     // long services stay covered

let updateState = { status: 'idle', version: null, percent: 0, current: app.getVersion() };
let fakeTimer = null;

function pushUpdateState(patch) {
  updateState = { ...updateState, ...patch };
  if (controlWin && !controlWin.isDestroyed()) {
    controlWin.webContents.send('itb:update', updateState);
  }
}

function checkForUpdate() {
  // Never disturb a download / a downloaded update that is waiting.
  if (updateState.status === 'downloading' || updateState.status === 'ready') return;
  autoUpdater.checkForUpdates().catch((err) => {
    console.error('[itb] update check failed:', err && err.message);
  });
}

function startUpdateDownload() {
  if (FAKE_UPDATE) {
    let pct = 0;
    pushUpdateState({ status: 'downloading', percent: 0 });
    clearInterval(fakeTimer);
    fakeTimer = setInterval(() => {
      pct += 10;
      if (pct >= 100) {
        clearInterval(fakeTimer);
        pushUpdateState({ status: 'ready', percent: 100 });
      } else {
        pushUpdateState({ status: 'downloading', percent: pct });
      }
    }, 400);
    return;
  }
  pushUpdateState({ status: 'downloading', percent: 0 });
  autoUpdater.downloadUpdate().catch((err) => {
    console.error('[itb] update download failed:', err && err.message);
    pushUpdateState({ status: 'error', message: 'The download did not finish. Check your internet connection and try again.' });
  });
}

function installUpdateNow() {
  if (FAKE_UPDATE) {
    console.log('[itb] (fake update) would restart and install now');
    pushUpdateState({ status: 'idle', version: null, percent: 0 });
    return;
  }
  // Stop the Python backend first: the installer cannot replace
  // main_fixed.exe while it is still running. isQuitting = true keeps the
  // "backend stopped unexpectedly" error box from appearing.
  isQuitting = true;
  stopBackend();
  setTimeout(() => autoUpdater.quitAndInstall(true, true), 800);
}

function setupAutoUpdater() {
  if (FAKE_UPDATE) {
    console.log('[itb] FAKE update mode — popup will appear in ~6s');
    setTimeout(() => {
      const [a, b, c] = app.getVersion().split('.').map((n) => parseInt(n, 10) || 0);
      pushUpdateState({ status: 'available', version: `${a}.${b}.${c + 1}`, percent: 0 });
    }, 6000);
    return;
  }
  if (!app.isPackaged) {
    console.log('[itb] auto-update skipped (development build)');
    return;
  }

  autoUpdater.autoDownload = false;          // ask first, then download
  autoUpdater.autoInstallOnAppQuit = false;  // "Later" means later

  autoUpdater.on('update-available', (info) => {
    console.log(`[itb] update available: ${info.version}`);
    pushUpdateState({ status: 'available', version: info.version, percent: 0 });
  });
  autoUpdater.on('update-not-available', () => console.log('[itb] app is up to date'));
  autoUpdater.on('download-progress', (p) => {
    pushUpdateState({ status: 'downloading', percent: Math.round(p.percent || 0) });
  });
  autoUpdater.on('update-downloaded', (info) => {
    console.log(`[itb] update downloaded: ${info.version}`);
    pushUpdateState({ status: 'ready', version: info.version, percent: 100 });
  });
  autoUpdater.on('error', (err) => {
    // Offline / no release yet is normal — only surface it mid-download.
    console.error('[itb] updater error:', err && err.message);
    if (updateState.status === 'downloading') {
      pushUpdateState({ status: 'error', message: 'The download did not finish. Check your internet connection and try again.' });
    }
  });

  setTimeout(checkForUpdate, UPDATE_CHECK_DELAY_MS);
  setInterval(checkForUpdate, UPDATE_CHECK_EVERY_MS);
}

// ─────────────────────────── IPC surface ─────────────────────────────
function registerIpc() {
  ipcMain.handle('itb:update-state', () => updateState);
  ipcMain.handle('itb:update-download', () => { startUpdateDownload(); return { ok: true }; });
  ipcMain.handle('itb:update-install', () => { installUpdateNow(); return { ok: true }; });

  ipcMain.handle('itb:list-displays', () => ({
    desktop: true,
    electron: true,
    monitors: listDisplays(),
    target: getTarget(),
    open: !!(outputWin && !outputWin.isDestroyed()),
  }));

  ipcMain.handle('itb:get-target', () => ({ screen_index: getTarget() }));

  ipcMain.handle('itb:set-target', (_e, idx) => {
    const saved = setTarget(idx);
    const mons = listDisplays();
    const label = (saved >= 0 && saved < mons.length) ? mons[saved].label : null;

    // If the output is live, relocate it to the newly chosen monitor.
    if (outputWin && !outputWin.isDestroyed()) {
      const churchId = outputWin.__churchId;
      destroyOutputWindow();
      createOutputWindow(churchId);
    }
    return { ok: true, screen_index: saved, label };
  });

  ipcMain.handle('itb:output-on', (_e, churchId) => {
    const res = createOutputWindow(churchId);
    if (outputWin) outputWin.__churchId = churchId;
    return res;
  });

  ipcMain.handle('itb:output-off', () => destroyOutputWindow());

  ipcMain.handle('itb:output-status', () => ({
    desktop: true,
    electron: true,
    open: !!(outputWin && !outputWin.isDestroyed()),
    screen_index: getTarget(),
  }));

  ipcMain.handle('itb:mic-status', async () => ({
    status: await ensureMicrophoneAccess(),
    platform: process.platform,
  }));

  ipcMain.handle('itb:open-mic-settings', () => {
    if (process.platform === 'win32') shell.openExternal('ms-settings:privacy-microphone');
    else if (process.platform === 'darwin') {
      shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone');
    }
    return { ok: true };
  });
}

// Rebuild/relocate output when monitors are plugged in or unplugged.
function watchDisplays() {
  const relocate = () => {
    if (!outputWin || outputWin.isDestroyed()) return;
    const churchId = outputWin.__churchId;
    destroyOutputWindow();
    setTimeout(() => createOutputWindow(churchId), 400);
  };
  screen.on('display-added', relocate);
  screen.on('display-removed', relocate);
  screen.on('display-metrics-changed', relocate);
}

// ─────────────────────────── lifecycle ───────────────────────────────
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (controlWin) {
      if (controlWin.isMinimized()) controlWin.restore();
      controlWin.focus();
    }
  });

  app.whenReady().then(async () => {
    configurePermissions();
    registerIpc();
    await ensureMicrophoneAccess();

    const ok = await startBackend();
    if (!ok) { app.quit(); return; }

    createControlWindow();
    watchDisplays();
    setupAutoUpdater();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createControlWindow();
    });
  });

  app.on('before-quit', () => { isQuitting = true; stopBackend(); });
  app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
  process.on('exit', stopBackend);
}