[CmdletBinding()]
param(
    [string[]]$Checks = @('contract','blackbox','branch','mutation','snapshot','fuzz','dependency','cross_env')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = 'D:\hermes'
$project = Join-Path $root 'codex_projects\codex-memory-platform'
$adapter = Join-Path $project 'tools\production_gate_adapter.py'
$python = Join-Path $root '.venv\Scripts\python.exe'

$guard = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'codex_native_process_guard.ps1') 2>&1
if ($LASTEXITCODE -ne 0) { throw "Native process guard failed: $($guard -join "`n")" }
$env:CODEX_MEMORY_CPU_THREADS = '2'
$env:CODEX_MEMORY_MAX_LENGTH = '128'
$output = @(& $python -u $adapter @Checks 2>&1)
$code = $LASTEXITCODE
$text = ($output | ForEach-Object { [string]$_ }) -join "`n"
if ($code -ne 0) {
    Write-Output $text
    Write-Output "CODEX_GATE_EXIT=$code"
    exit $code
}
$text
