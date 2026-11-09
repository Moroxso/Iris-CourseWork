[CmdletBinding(DefaultParameterSetName = 'Single')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Single')][string]$Prompt,
    [Parameter(Mandatory = $true, ParameterSetName = 'Single')]
    [Parameter(ParameterSetName = 'Interactive')][string]$Out,
    [Parameter(Mandatory = $true, ParameterSetName = 'Interactive')][switch]$Interactive,
    [Parameter(ParameterSetName = 'Interactive')][switch]$Live,
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$Snapshot,
    [Parameter(Mandatory = $true)][string]$Checkpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$taskRepo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskPython = Join-Path $taskRepo '.venv/Scripts/python.exe'
$taskKitScript = Join-Path $PSScriptRoot 'coursework_offline_kit.py'
$taskDemoScript = Join-Path $PSScriptRoot 'coursework_trained_demo.py'
$taskManifest = [IO.Path]::GetFullPath($Manifest)
$taskCheckpoint = [IO.Path]::GetFullPath($Checkpoint)

if ($Interactive) {
    $taskConsoleArgs = @('-I', '-B', (Join-Path $PSScriptRoot 'coursework_console.py'),
        '--manifest', $taskManifest, '--snapshot', $Snapshot, '--checkpoint', $taskCheckpoint)
    if ($Out) { $taskConsoleArgs += @('--out', $Out) }
    if ($Live) { $taskConsoleArgs += '--live' }
    & $taskPython @taskConsoleArgs
    if ($LASTEXITCODE -ne 0) { throw 'Console stopped with an error; retain the session output.' }
    return
}

if (-not [IO.Path]::IsPathRooted($Out) -or (Test-Path -LiteralPath $Out)) {
    throw 'Out must be a fresh absolute directory.'
}
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'The existing project venv Python is missing; nothing will be installed.'
}

& $taskPython -I -B $taskKitScript verify --manifest $taskManifest --snapshot $Snapshot --checkpoint $taskCheckpoint
if ($LASTEXITCODE -ne 0) {
    throw 'Offline kit verification failed. No model inference was launched.'
}
& $taskPython -I -B $taskDemoScript --snapshot $Snapshot --checkpoint $taskCheckpoint --out $Out --prompt $Prompt
if ($LASTEXITCODE -ne 0) {
    throw 'Demo did not complete. Retain partial outputs and inspect error.json.'
}
Write-Host ('Saved result and offline viewer: ' + (Join-Path $Out 'viewer.html'))
