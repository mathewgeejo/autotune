$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.tools/python'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.tools/uv-cache'
$uv = Join-Path $PSScriptRoot '.tools/uv/uv.exe'
if (!(Test-Path $uv)) {
    New-Item -ItemType Directory -Force '.tools/uv' | Out-Null
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools/uv.zip'
    Expand-Archive '.tools/uv.zip' '.tools/uv' -Force
    Remove-Item -LiteralPath '.tools/uv.zip'
}
& $uv python install 3.10
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
if (!(Test-Path '.venv/Scripts/python.exe')) {
    & $uv venv --python 3.10 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
}
$requirements = if (Test-Path 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.txt' }
& $uv pip install --python .venv/Scripts/python.exe -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
& .venv/Scripts/python.exe scripts/install_assets.py
if ($LASTEXITCODE -ne 0) { throw 'Audio asset installation failed. Run setup again to resume.' }
# Preserve the installer cache so a later repair does not re-download packages.
Write-Host 'Ready. Double-click launch.cmd to open Voice Studio.'
