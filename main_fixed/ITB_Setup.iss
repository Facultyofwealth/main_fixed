; ============================================================
;  In the Beginning AI — Inno Setup script
;  Wraps the PyInstaller-built exe in a proper Windows installer
;  with a wizard UI, Start Menu shortcut, optional desktop icon,
;  and a clean uninstaller entry in "Apps & Features".
;
;  Requires Inno Setup (free): https://jrsoftware.org/isinfo.php
;  Open this file in the Inno Setup Compiler (or run ISCC.exe
;  from the command line) to produce ITB-Setup.exe.
; ============================================================

#define MyAppName "In the Beginning AI"
#define MyAppVersion "1.0.1"
#define MyAppPublisher "Faculty of Wealth"
#define MyAppURL "https://inthebeginning.ai"
#define MyAppExeName "start_in_the_beginning.exe"

[Setup]
AppId={{8F1B6C2E-4A3D-4E7B-9C2A-ITBAPPGUID01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Installer's own icon (shown on the Setup.exe file itself and in the wizard)
SetupIconFile=icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
OutputDir=installer_output
OutputBaseFilename=ITB-Setup
; Uninstall entry in "Apps & Features" also uses this icon
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The PyInstaller-built exe — adjust the Source path if your dist folder
; layout differs. This is the ONLY required payload since PyInstaller
; already bundled kjv.json, the HTML UI, and all Python deps into it.
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
