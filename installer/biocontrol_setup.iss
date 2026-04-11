; =============================================================================
; Biocontrol Dashboard - Inno Setup Installer Script
; =============================================================================
; Prerequisites:
;   - Inno Setup 6.x  (https://jrsoftware.org/isinfo.php)
;   - installer\dependencies\python-3.10.9-amd64.exe      (download separately)
;   - installer\dependencies\dotnet-runtime-8.0-win-x64.exe (download separately)
;   - installer\dependencies\DWSIM\* (copy from DWSIM install)
; Compile:
;   iscc /DAppVersion=1.0.0 biocontrol_setup.iss
; =============================================================================

#define AppName        "Biocontrol Dashboard"
#define AppVersion     "1.0.0"
#define AppPublisher   "César Augusto García Echeverry"
#define AppURL         "https://github.com/CesarGarech/Biocontrol_modeling_project"
#define AppExeName     "run_dashboard.bat"

[Setup]
AppId={{B7C2D4E1-8F3A-4B6C-9D0E-1A2B3C4D5E6F}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\BiocontrolDashboard
DefaultGroupName={#AppName}
AllowNoIcons=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0.17763
PrivilegesRequired=admin
OutputDir=Output
OutputBaseFilename=BiocontrolDashboard-Setup-v{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[CustomMessages]
english.WelcomeLabel2=This wizard will install [name/ver] on your computer.%n%nThe following components will be installed automatically:%n  - .NET Runtime 8.0%n  - Python 3.10.9 (if not already present)%n  - DWSIM process simulator%n  - All required Python libraries%n%nClick Next to continue.
spanish.WelcomeLabel2=Este asistente instalará [name/ver] en su computador.%n%nSe instalarán automáticamente los siguientes componentes:%n  - .NET Runtime 8.0%n  - Python 3.10.9 (si no está instalado)%n  - Simulador de procesos DWSIM%n  - Todas las librerías Python necesarias%n%nHaga clic en Siguiente para continuar.

english.InstallingLibraries=Installing Python libraries, please wait...
spanish.InstallingLibraries=Instalando librerías Python, por favor espere...

[Tasks]
Name: "desktopicon";     Description: "{cm:CreateDesktopIcon}";          GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "launchafterdone"; Description: "Launch Biocontrol Dashboard now";  GroupDescription: "After installation:";  Flags: unchecked

[Files]
; .NET Runtime 8.0 installer
Source: "..\dependencies\dotnet-sdk-8.0.419-win-x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsDotNet8Installed

; Python 3.10.9 installer
Source: "..\dependencies\python-3.10.9-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsPython310Installed

; DWSIM files - Preprocessor directive checks for folder during COMPILATION
#if exist("..\dependencies\DWSIM")
Source: "..\dependencies\DWSIM\*"; DestDir: "{localappdata}\DWSIM"; Flags: recursesubdirs createallsubdirs ignoreversion
#endif

; Project root files
Source: "..\main.py";             DestDir: "{app}"; Flags: ignoreversion
Source: "..\Menu.py";             DestDir: "{app}"; Flags: ignoreversion
Source: "..\theory.py";           DestDir: "{app}"; Flags: ignoreversion
Source: "..\style.css";           DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt";    DestDir: "{app}"; Flags: ignoreversion
Source: "..\setup.py";            DestDir: "{app}"; Flags: ignoreversion
Source: "..\run_dashboard.bat";   DestDir: "{app}"; Flags: ignoreversion
Source: "..\verify_python.bat";   DestDir: "{app}"; Flags: ignoreversion
Source: "..\version.py";          DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md";        DestDir: "{app}"; Flags: ignoreversion

; Project directories (recursive)
Source: "..\Body\*";        DestDir: "{app}\Body";        Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\Utils\*";       DestDir: "{app}\Utils";       Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\Data\*";        DestDir: "{app}\Data";        Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\Simulation\*";  DestDir: "{app}\Simulation";  Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\Images\*";      DestDir: "{app}\Images";      Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\References\*";  DestDir: "{app}\References";  Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\Examples\*";    DestDir: "{app}\Examples";    Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\test_data\*";   DestDir: "{app}\test_data";   Flags: recursesubdirs createallsubdirs ignoreversion

; Post-install batch script
Source: "post_install.bat"; DestDir: "{app}"; Flags: ignoreversion deleteafterinstall

[Icons]
Name: "{group}\{#AppName}";            Filename: "{app}\run_dashboard.bat"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\run_dashboard.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\post_install.bat"; \
    Description: "{cm:InstallingLibraries}"; \
    Parameters: """{app}"""; \
    WorkingDir: "{app}"; \
    Flags: runhidden waituntilterminated; \
    StatusMsg: "{cm:InstallingLibraries}"

Filename: "{app}\run_dashboard.bat"; \
    Description: "Launch {#AppName}"; \
    WorkingDir: "{app}"; \
    Flags: nowait postinstall skipifsilent; \
    Tasks: launchafterdone

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\Output"

[Code]

// ---------------------------------------------------------------------------
// IsDotNet8Installed
// Verifica si .NET 8 Desktop Runtime (x64) está instalado mediante el registro
// ---------------------------------------------------------------------------
function IsDotNet8Installed: Boolean;
begin
  // Revisa la clave oficial del registro de .NET para la versión 8.0 x64
  Result := RegKeyExists(HKLM, 'SOFTWARE\dotnet\Setup\InstalledVersions\x64\sharedfx\Microsoft.WindowsDesktop.App\8.0') or
            RegKeyExists(HKLM, 'SOFTWARE\WOW6432Node\dotnet\Setup\InstalledVersions\x64\sharedfx\Microsoft.WindowsDesktop.App\8.0');
end;

// ---------------------------------------------------------------------------
// InstallDotNet
// Ejecuta el instalador de .NET 8 silenciosamente si no está instalado
// ---------------------------------------------------------------------------
procedure InstallDotNet;
var
  InstallerPath: String;
  ResultCode: Integer;
begin
  if IsDotNet8Installed then
  begin
    Log('.NET 8 Runtime already installed — skipping.');
    Exit;
  end;

  InstallerPath := ExpandConstant('{tmp}\dotnet-runtime-8.0-win-x64.exe');
  Log('Installing .NET 8 from: ' + InstallerPath);

  if not FileExists(InstallerPath) then
  begin
    MsgBox('.NET 8 installer not found at:' + #13#10 + InstallerPath + #13#10#13#10 +
           'Please ensure the installer was bundled correctly.', mbError, MB_OK);
    Exit;
  end;

  // Ejecución silenciosa para el instalador de Microsoft (/install /quiet /norestart)
  if not Exec(InstallerPath, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    MsgBox('.NET 8 installation failed with code: ' + IntToStr(ResultCode), mbError, MB_OK);
  end
  else
  begin
    Log('.NET 8 installed successfully (exit code ' + IntToStr(ResultCode) + ').');
  end;
end;

// ---------------------------------------------------------------------------
// IsPython310Installed
// Checks HKLM and HKCU for any Python 3.10.x installation.
// ---------------------------------------------------------------------------
function IsPython310Installed: Boolean;
var
  SubKey: String;
  InstalledPath: String;
begin
  Result := False;

  SubKey := 'SOFTWARE\Python\PythonCore\3.10\InstallPath';
  if RegQueryStringValue(HKLM, SubKey, '', InstalledPath) then
  begin
    if InstalledPath <> '' then
    begin
      Result := True;
      Exit;
    end;
  end;

  if RegQueryStringValue(HKCU, SubKey, '', InstalledPath) then
  begin
    if InstalledPath <> '' then
    begin
      Result := True;
      Exit;
    end;
  end;

  SubKey := 'SOFTWARE\WOW6432Node\Python\PythonCore\3.10\InstallPath';
  if RegQueryStringValue(HKLM, SubKey, '', InstalledPath) then
  begin
    if InstalledPath <> '' then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

// ---------------------------------------------------------------------------
// InstallPython
// Runs the bundled Python 3.10.9 installer silently.
// ---------------------------------------------------------------------------
procedure InstallPython;
var
  InstallerPath: String;
  ResultCode: Integer;
begin
  if IsPython310Installed then
  begin
    Log('Python 3.10 already installed — skipping.');
    Exit;
  end;

  InstallerPath := ExpandConstant('{tmp}\python-3.10.9-amd64.exe');
  Log('Installing Python 3.10.9 from: ' + InstallerPath);

  if not FileExists(InstallerPath) then
  begin
    MsgBox('Python 3.10.9 installer not found at:' + #13#10 + InstallerPath + #13#10#13#10 +
           'Please ensure the installer was bundled correctly.', mbError, MB_OK);
    Exit;
  end;

  if not Exec(InstallerPath, '/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1 Include_test=0', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    MsgBox('Python 3.10.9 installation failed with code: ' + IntToStr(ResultCode), mbError, MB_OK);
  end
  else
  begin
    Log('Python 3.10.9 installed successfully (exit code ' + IntToStr(ResultCode) + ').');
  end;
end;

// ---------------------------------------------------------------------------
// CurStepChanged event hook
// Installs .NET y Python antes de copiar los archivos.
// ---------------------------------------------------------------------------
procedure CurStepChanged(CurStep: TSetupStep);
begin
  case CurStep of
    ssInstall:
      begin
        InstallDotNet;
        InstallPython;
      end;
  end;
end;