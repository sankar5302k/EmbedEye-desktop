# EmbedEye Desktop App Self-Contained Build Script (win-x64)
# Produces standalone distribution in dist/EmbedEye/

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Building EmbedEye Desktop (Self-Contained win-x64)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$RepoRoot = $PSScriptRoot
$ProjectFile = Join-Path $RepoRoot "gui\EmbedEye.Gui\EmbedEye.Gui.csproj"
$OutputDir = Join-Path $RepoRoot "dist\EmbedEye"

if (Test-Path "C:\Users\Admin\.dotnet\dotnet.exe") {
    $env:DOTNET_ROOT = "C:\Users\Admin\.dotnet"
    $env:PATH = "C:\Users\Admin\.dotnet;" + $env:PATH
}

Write-Host "Publishing project: $ProjectFile" -ForegroundColor Yellow
Write-Host "Target output dir: $OutputDir" -ForegroundColor Yellow

& dotnet publish $ProjectFile `
    -c Release `
    -r win-x64 `
    --self-contained true `
    -p:PublishSingleFile=false `
    -o $OutputDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] EmbedEye self-contained build completed successfully!" -ForegroundColor Green
    Write-Host "Executable located at: $OutputDir\EmbedEye.Gui.exe" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
