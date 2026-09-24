'use strict';
/**
 * Bridges the renderer (In_the_Beginning.html / display page) to the main
 * process. contextIsolation stays ON — the page gets a narrow, explicit API
 * surface via window.itbDesktop and nothing else.
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('itbDesktop', {
  isElectron: true,
  platform: process.platform,

  // Monitors / output target
  listDisplays: () => ipcRenderer.invoke('itb:list-displays'),
  getTarget: () => ipcRenderer.invoke('itb:get-target'),
  setTarget: (index) => ipcRenderer.invoke('itb:set-target', index),

  // Output window control
  outputOn: (churchId) => ipcRenderer.invoke('itb:output-on', churchId),
  outputOff: () => ipcRenderer.invoke('itb:output-off'),
  outputStatus: () => ipcRenderer.invoke('itb:output-status'),

  // Microphone (OS-level, not a browser prompt)
  micStatus: () => ipcRenderer.invoke('itb:mic-status'),
  openMicSettings: () => ipcRenderer.invoke('itb:open-mic-settings'),

  // Auto-update (popup is drawn by the page; main process does the work)
  getUpdateState: () => ipcRenderer.invoke('itb:update-state'),
  downloadUpdate: () => ipcRenderer.invoke('itb:update-download'),
  installUpdate: () => ipcRenderer.invoke('itb:update-install'),
  onUpdate: (callback) => {
    const handler = (_event, state) => callback(state);
    ipcRenderer.on('itb:update', handler);
    return () => ipcRenderer.removeListener('itb:update', handler);
  },
});