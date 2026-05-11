; =============================================================================
; Biocontrol Dashboard - Inno Setup Installer Script
; =============================================================================
; Prerequisites:
;   - Inno Setup 6.x  (https://jrsoftware.org/isinfo.php)
;   - installer\dependencies\python-3.10.9-amd64.exe      (download separately)
;   - installer\dependencies\dotnet-sdk-8.0.x-win-x64.exe (download separately)
;   - installer\dependencies\OllamaSetup.exe              (download separately)
;   - installer\dependencies\DWSIM\* (copy from DWSIM install)
; Compile:
;   iscc /DAppVersion=1.0.1 biocontrol_setup.iss
; =============================================================================

#define AppName        "Biocontrol Dashboard"
#define AppVersion     "1.0.2"
#define AppPublisher   "Cesar Augusto Garcia Echeverry"
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

[CustomMessages]
english.WelcomeLabel2=This wizard will install [name/ver] on your computer.%n%nThe following components will be installed automatically:%n  - .NET Runtime 8.0 or higher%n  - Python 3.10.x%n  - DWSIM process simulator%n  - Ollama (Local LLM engine)%n  - All required Python libraries%n%nClick Next to continue.
english.InstallingLibraries=Installing Python libraries and starting services, please wait...

[Tasks]
Name: "desktopicon";     Description: "{cm:CreateDesktopIcon}";          GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "launchafterdone"; Description: "Launch Biocontrol Dashboard now";  GroupDescription: "After installation:";  Flags: unchecked

[Files]
; .NET Runtime 8.0 installer
Source: "..\dependencies\dotnet-sdk-8.0.419-win-x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsDotNet8OrHigherInstalled

; Python 3.10.9 installer
Source: "..\dependencies\python-3.10.9-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsPython310Installed

; Ollama installer
Source: "..\dependencies\OllamaSetup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not IsOllamaInstalled

; DWSIM files - Preprocessor directive checks for folder during COMPILATION
#if FileExists("..\dependencies\DWSIM\DWSIM.Automation.dll")
Source: "..\dependencies\DWSIM\*"; DestDir: "{pf}\DWSIM"; Flags: recursesubdirs createallsubdirs ignoreversion
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
Name: "{autodesktop}\{#AppName}";     Filename: "{app}\run_dashboard.bat"; WorkingDir: "{app}"; Tasks: desktopicon
; DWSIM Desktop Shortcut
Name: "{autodesktop}\DWSIM";          Filename: "{pf}\DWSIM\DWSIM.exe"; WorkingDir: "{pf}\DWSIM"; Tasks: desktopicon

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
Type: filesandordirs; Name: "{localappdata}\BiocontrolDashboard\.venv"
Type: filesandordirs; Name: "{localappdata}\BiocontrolDashboard"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: filesandordirs; Name: "{app}\Output"

[Code]

function IsDotNet8OrHigherInstalled: Boolean;
var
  Versions: TArrayOfString;
  I, MajorVersion, PosDot: Integer;
  BaseKey: String;
begin
  Result := False;
  BaseKey := 'SOFTWARE\dotnet\Setup\InstalledVersions\x64\sharedfx\Microsoft.WindowsDesktop.App';
  if RegGetSubkeyNames(HKLM, BaseKey, Versions) then
  begin
    for I := 0 to GetArrayLength(Versions) - 1 do
    begin
      PosDot := Pos('.', Versions[I]);
      if PosDot > 0 then
      begin
        MajorVersion := StrToIntDef(Copy(Versions[I], 1, PosDot - 1), 0);
        if MajorVersion >= 8 then begin Result := True; Exit; end;
      end;
    end;
  end;
  BaseKey := 'SOFTWARE\WOW6432Node\dotnet\Setup\InstalledVersions\x64\sharedfx\Microsoft.WindowsDesktop.App';
  if RegGetSubkeyNames(HKLM, BaseKey, Versions) then
  begin
    for I := 0 to GetArrayLength(Versions) - 1 do
    begin
      PosDot := Pos('.', Versions[I]);
      if PosDot > 0 then
      begin
        MajorVersion := StrToIntDef(Copy(Versions[I], 1, PosDot - 1), 0);
        if MajorVersion >= 8 then begin Result := True; Exit; end;
      end;
    end;
  end;
end;

procedure InstallDotNet;
var InstallerPath: String; ResultCode: Integer;
begin
  if IsDotNet8OrHigherInstalled then Exit;
  InstallerPath := ExpandConstant('{tmp}\dotnet-sdk-8.0.419-win-x64.exe');
  if FileExists(InstallerPath) then
    Exec(InstallerPath, '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

function IsPython310Installed: Boolean;
var SubKey, InstalledPath: String;
begin
  Result := False;
  SubKey := 'SOFTWARE\Python\PythonCore\3.10\InstallPath';
  if RegQueryStringValue(HKLM, SubKey, '', InstalledPath) and (InstalledPath <> '') then begin Result := True; Exit; end;
  if RegQueryStringValue(HKCU, SubKey, '', InstalledPath) and (InstalledPath <> '') then begin Result := True; Exit; end;
  SubKey := 'SOFTWARE\WOW6432Node\Python\PythonCore\3.10\InstallPath';
  if RegQueryStringValue(HKLM, SubKey, '', InstalledPath) and (InstalledPath <> '') then begin Result := True; Exit; end;
end;

procedure InstallPython;
var InstallerPath: String; ResultCode: Integer;
begin
  if IsPython310Installed then Exit;
  InstallerPath := ExpandConstant('{tmp}\python-3.10.9-amd64.exe');
  if FileExists(InstallerPath) then
    Exec(InstallerPath, '/passive InstallAllUsers=1 TargetDir="' + ExpandConstant('{pf}\Python310') + '" PrependPath=1 Include_pip=1 Include_test=0', '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
end;

// ---------------------------------------------------------------------------
// Ollama Installation Logic
// ---------------------------------------------------------------------------
function IsOllamaInstalled: Boolean;
begin
  // Standard Ollama installation path
  Result := FileExists(ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe'));
end;

procedure InstallOllama;
var InstallerPath: String; ResultCode: Integer;
begin
  if IsOllamaInstalled then
  begin
    Log('Ollama is already installed - skipping.');
    Exit;
  end;

  InstallerPath := ExpandConstant('{tmp}\OllamaSetup.exe');
  Log('Installing Ollama from: ' + InstallerPath);
  if FileExists(InstallerPath) then
  begin
    // Ollama installation is usually silent by default with /SILENT
    Exec(InstallerPath, '/SILENT', '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  case CurStep of
    ssPostInstall:
      begin
        InstallDotNet;
        InstallPython;
        InstallOllama;
      end;
  end;
end;