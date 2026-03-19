param(
    [string]$Python = "python",
    [string]$NcnnToolsDir,
    [string]$Workspace = "work",
    [string]$CalibDir,
    [string]$CalibList,
    [int]$ImgSz = 640,
    [int]$DemoCalibrationRepeats = 8,
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $Root "requirements.txt"
$ScriptPath = Join-Path $Root "quantize_yolov8n_ncnn_int8.py"

if (-not (Test-Path $VenvDir)) {
    & $Python -m venv $VenvDir
}

& $PythonExe -m pip install -r $Requirements

$ArgsList = @($ScriptPath, "--workspace", $Workspace, "--imgsz", $ImgSz, "--demo-calibration-repeats", $DemoCalibrationRepeats)

if ($NcnnToolsDir) {
    $ArgsList += @("--ncnn-tools-dir", $NcnnToolsDir)
}

if ($CalibDir) {
    $ArgsList += @("--calib-dir", $CalibDir)
}

if ($CalibList) {
    $ArgsList += @("--calib-list", $CalibList)
}

if ($ExtraArgs) {
    $ArgsList += $ExtraArgs
}

& $PythonExe @ArgsList
