[CmdletBinding()]
param(
    [int]$Attempts = 3,
    [double]$TimeoutSeconds = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = 'D:\hermes'
$project = Join-Path $root 'codex_projects\codex-memory-platform'
$python = Join-Path $root '.venv\Scripts\python.exe'
$probe = Join-Path $project 'tools\codegraph_mcp_health.py'
$state = Join-Path $root 'memory_store\codex\codegraph_health.json'
$lockPath = Join-Path $root 'memory_store\codex\codegraph_health.lock'
$handle = $null
try {
    $parent = Split-Path -Parent $state
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    try {
        $handle = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch [IO.IOException] {
        throw 'CodeGraph health probe is already running.'
    }
    $output = @(& $python -u $probe '--attempts' ([string]$Attempts) '--timeout' ([string]$TimeoutSeconds) 2>&1)
    $code = $LASTEXITCODE
    $text = ($output | ForEach-Object { [string]$_ }) -join "`n"
    if ($code -ne 0) {
        [IO.File]::WriteAllText($state, $text, [Text.UTF8Encoding]::new($false))
        Write-Output $text
        exit $code
    }
    [IO.File]::WriteAllText($state, $text, [Text.UTF8Encoding]::new($false))
    $text
} finally {
    if ($handle) { $handle.Dispose() }
}
