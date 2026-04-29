# build-release.ps1 — Build optimized release archive for ai-gateway
$ErrorActionPreference = "Stop"
$name = "ai-gateway"
$bin  = "target/release/$name.exe"
$out  = "release"
$assetsOut = "$out/Assets"

Write-Host "Building optimized tray-enabled release binary..." -ForegroundColor Cyan
cargo build --release -p $name --features tray
if ($LASTEXITCODE -ne 0) { throw "Cargo build failed" }

if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Path $out | Out-Null
New-Item -ItemType Directory -Path $assetsOut | Out-Null

Copy-Item $bin                              "$out/$name.exe"
Copy-Item "crates/$name/README.md"          "$out/README.md"
Copy-Item "crates/$name/config.example.yaml" "$out/config.example.yaml"
Copy-Item "crates/$name/config.example.yaml" "$out/config.yaml"
Copy-Item "Assets/icon.ico"                 "$assetsOut/icon.ico"
Copy-Item "Assets/logo.jpg"                 "$assetsOut/logo.jpg"

$hash = (Get-FileHash "$out/$name.exe" -Algorithm SHA256).Hash.ToLower()
"$hash  $name.exe" | Set-Content "$out/SHA256SUMS.txt" -NoNewline
Write-Host "SHA256: $hash" -ForegroundColor Green

Compress-Archive -Path "$out/*" -DestinationPath "$out/$name-windows-x86_64.zip" -Force
Write-Host "Release archive: $out/$name-windows-x86_64.zip" -ForegroundColor Green
