# build-installer.ps1 — Build release + compile Inno Setup installer
# Requires: Inno Setup 6+ with iscc.exe on PATH or in default install location
$ErrorActionPreference = "Stop"

# 1. Build the release bundle
Write-Host "Step 1: Building release..." -ForegroundColor Cyan
& "$PSScriptRoot\build-release.ps1"

# 2. Locate iscc.exe
$iscc = $null
$found = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
if ($found) {
    $iscc = $found.Source
}
if (-not $iscc) {
    # Check all fixed drives for Inno Setup 6
    $drives = (Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ne $null }).Root
    $candidates = foreach ($drv in $drives) {
        "${drv}Program Files (x86)\Inno Setup 6\ISCC.exe"
        "${drv}Program Files\Inno Setup 6\ISCC.exe"
    }
    foreach ($p in $candidates) {
        if (Test-Path $p) { $iscc = $p; break }
    }
}
if (-not $iscc) {
    Write-Error "Inno Setup compiler (iscc.exe) not found. Install from https://jrsoftware.org/isinfo.php"
    exit 1
}
Write-Host "Found ISCC: $iscc" -ForegroundColor Gray

# 3. Compile installer
$issFile = "$PSScriptRoot\installer.iss"
Write-Host "Step 2: Compiling installer with $iscc ..." -ForegroundColor Cyan
& $iscc $issFile
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }

$version = "0.1.0"
$output  = "release\OBEY-API-Gateway-Setup-$version.exe"
if (Test-Path $output) {
    $hash = (Get-FileHash $output -Algorithm SHA256).Hash.ToLower()
    Write-Host "Installer: $output" -ForegroundColor Green
    Write-Host "SHA256:    $hash" -ForegroundColor Green
} else {
    Write-Warning "Expected output not found at $output"
}
