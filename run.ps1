# Transcribes a recording from .\audio into .\transcribed, writing both the
# Urdu-script text and the Roman Urdu version.
#
#   .\run.ps1 audio.ogg              # exact name, extension optional
#   .\run.ps1 audio                  # partial name is enough
#   .\run.ps1 audio.ogg --language urdu
#   .\run.ps1                        # no name: lists files in .\audio

param(
    [string]$Name,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Extra
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Keep Urdu text from being mangled by the console code page.
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$audioDir = Join-Path $PSScriptRoot 'audio'
$outDir = Join-Path $PSScriptRoot 'transcribed'

if (-not (Test-Path -LiteralPath $audioDir)) {
    Write-Host "Folder not found: $audioDir" -ForegroundColor Red
    exit 1
}

if (-not $Name) {
    Write-Host 'Usage: .\run.ps1 "<file name in .\audio>" [extra transcribe.py flags]' -ForegroundColor Yellow
    Write-Host ''
    Write-Host 'Available files:' -ForegroundColor Yellow
    Get-ChildItem -LiteralPath $audioDir -File | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

# Allow the name without its extension.
$wanted = $Name.Trim()
if (-not [IO.Path]::GetExtension($wanted)) { $wanted += '.ogg' }

$file = Get-ChildItem -LiteralPath $audioDir -File |
    Where-Object { $_.Name -eq $wanted } |
    Select-Object -First 1

# Fall back to a partial name match.
if (-not $file) {
    $file = Get-ChildItem -LiteralPath $audioDir -File |
        Where-Object { $_.Name -like "*$wanted*" } |
        Select-Object -First 1
}

if (-not $file) {
    Write-Host "No match for '$Name' in $audioDir" -ForegroundColor Red
    Write-Host ''
    Write-Host 'Available files:' -ForegroundColor Yellow
    Get-ChildItem -LiteralPath $audioDir -File | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

Write-Host "Transcribing: $($file.Name)" -ForegroundColor Cyan

python transcribe.py $file.FullName --roman --outdir $outDir --format txt @Extra
$code = $LASTEXITCODE

if ($code -eq 0) {
    Write-Host ''
    Write-Host "Files written to: $outDir" -ForegroundColor Green
    Write-Host "  $($file.BaseName).txt        (Urdu script)"
    Write-Host "  $($file.BaseName).roman.txt  (Roman Urdu)"
} else {
    Write-Host ''
    Write-Host "Transcription failed (exit code $code)." -ForegroundColor Red
}

exit $code
